"""Alex: mixed supervised training + frozen-v1 corner distillation.

Initializes from a verified HF release, not a SECE optimizer checkpoint.
Training batches are exactly 50/50. Per-draw deterministic NumPy augmentation
and a saved sampler cursor permit mid-epoch restart without replaying updates.
SIGUSR1/SIGTERM request an atomic checkpoint at the next safe batch boundary.
Validation interrupted by a stop is rerun in full on resume before selection.
"""
import argparse
import copy
import hashlib
import json
import math
import random
import signal

import numpy as np
import torch
import torch.nn.functional as F
from easydict import EasyDict
from pathlib import Path
from safetensors.torch import load_file
from torch.utils.data import DataLoader

from mixed_utils import (
    PROJECT, GraphDataset, RoofNet, replace_batchnorm, forward_losses,
    save_checkpoint, worker_seed, BalancedBatchSampler, ReplayDataset,
    tagged_collate, inspect_data, digest, NORMALIZATION,
    validate, print_validation, selection_status, harmonic, domain_score, write_json,
)

STOP_REQUESTED = False


def request_stop(signum, frame):
    global STOP_REQUESTED
    STOP_REQUESTED = True


def should_stop():
    return STOP_REQUESTED


def initial_release(folder):
    names = ('model.safetensors', 'config.json', 'preprocessing.json', 'training_metadata.json')
    hashes = json.loads((folder / 'SHA256SUMS.json').read_text())
    identity = hashlib.sha256()
    for name in names:
        actual = digest(folder / name)
        if hashes.get(name) != actual:
            raise ValueError(f'Release hash mismatch: {name}')
        identity.update(f'{name}:{actual}\n'.encode())
    config = json.loads((folder / 'config.json').read_text())
    preprocessing = json.loads((folder / 'preprocessing.json').read_text())
    metadata = json.loads((folder / 'training_metadata.json').read_text())
    if config['model_type'] != 'koor_point2roof':
        raise ValueError('Expected Koor v1 release')
    if preprocessing['points'] != 1024 or preprocessing['normalization'] != NORMALIZATION:
        raise ValueError('Unsupported v1 preprocessing')
    return EasyDict(config['model_config']), preprocessing, metadata, load_file(str(folder / 'model.safetensors'), device='cpu'), identity.hexdigest()


def implementation_identity():
    paths = [Path(__file__), Path(__file__).with_name('mixed_utils.py'),
             PROJECT / 'v1/scripts/train_roofs.py', PROJECT / 'v1/scripts/train_corners.py',
             PROJECT / 'v1/scripts/evaluate_roofs.py']
    return {str(path.resolve().relative_to(PROJECT.resolve())): digest(path) for path in paths}


def fresh_cursor(epoch=0):
    return {'epoch': epoch, 'next_batch': 0, 'samples': 0, 'empty_batches': 0,
            'sums': {k: 0.0 for k in ('corner', 'refine', 'edge', 'kd_score', 'kd_offset', 'total')}}


def train_report(cursor):
    if not cursor['samples']:
        raise RuntimeError('No training samples in this epoch')
    report = {key: value / cursor['samples'] for key, value in cursor['sums'].items()}
    report.update(samples=cursor['samples'], synthetic_samples=cursor['samples'] // 2,
                  real_samples=cursor['samples'] // 2, empty_proposal_batches=cursor['empty_batches'])
    return report


# Distillation helpers below are shared with the previously tested SECE trainer.
def tensor_state_digest(module):
    result = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        result.update(name.encode())
        result.update(str((value.dtype, tuple(value.shape))).encode())
        result.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return result.hexdigest()

def teacher_predictions(teacher, points):
    if teacher.training or any(p.requires_grad for p in teacher.parameters()):
        raise RuntimeError("Teacher must remain frozen in eval mode")
    captured = {}

    def capture_logits(module, inputs, output):
        captured["logits"] = output.permute(0, 2, 1).squeeze(-1).detach()

    hook = teacher.cls_fc.register_forward_hook(capture_logits)
    try:
        with torch.no_grad():
            result = teacher({"points": points, "batch_size": len(points), "vectors": None})
            offsets = result["point_pred_offset"].detach() / teacher.model_cfg.PosRadius
    finally:
        hook.remove()
    return captured["logits"], offsets

def distillation_losses(student_logits, student_offsets, teacher_logits,
                        teacher_offsets, temperature):
    """Per-building normalized KD; teacher tensors never receive gradients.

    Logits: (B,N). Offsets: (B,N,3), in units of PosRadius, before radius scaling.
    KD score weights balance teacher positive/negative probability mass. Offset
    weights emphasize confident corners, avoiding background offset dominance.
    """
    teacher_logits = teacher_logits.detach()
    teacher_offsets = teacher_offsets.detach()
    if (student_logits.shape != teacher_logits.shape
            or student_offsets.shape != teacher_offsets.shape
            or student_offsets.shape != (*student_logits.shape, 3)):
        raise ValueError("Teacher/student per-point shapes do not match")
    for tensor in (student_logits, student_offsets, teacher_logits, teacher_offsets):
        if not torch.isfinite(tensor).all():
            raise RuntimeError("Non-finite distillation inputs")
    native_probability = teacher_logits.sigmoid()
    positive = native_probability / native_probability.sum(dim=1, keepdim=True).clamp_min(1e-8)
    negative = (1 - native_probability) / (1 - native_probability).sum(dim=1, keepdim=True).clamp_min(1e-8)
    weights = (positive + negative) * 0.5
    weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
    teacher_scaled = teacher_logits / temperature
    student_scaled = student_logits / temperature
    probability = teacher_scaled.sigmoid()
    kl = (probability * (F.logsigmoid(teacher_scaled) - F.logsigmoid(student_scaled))
          + (1 - probability) * (F.logsigmoid(-teacher_scaled) - F.logsigmoid(-student_scaled)))
    score = (weights * kl).sum(dim=1).mean() * temperature ** 2
    offset_weights = native_probability.square()
    offset_weights = offset_weights / offset_weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
    error = F.smooth_l1_loss(student_offsets, teacher_offsets, reduction="none", beta=0.1).mean(dim=-1)
    offset = (offset_weights * error).sum(dim=1).mean()
    return score, offset


def training_step(model, teacher, batch, optimizer, args):
    batch = {key: value.cuda(non_blocking=True) if torch.is_tensor(value) else value
             for key, value in batch.items()}
    mask = batch.pop('is_synthetic')
    size = batch['batch_size']
    if int(mask.sum()) * 2 != size:
        raise RuntimeError('Batch is not exactly 50% synthetic')
    optimizer.zero_grad(set_to_none=True)
    teacher_logits, teacher_offsets = teacher_predictions(teacher, batch['points'][mask])
    supervised, prediction, _, _ = forward_losses(model, batch)
    values = model.keypoint_det_net.train_dict
    kd_score, kd_offset = distillation_losses(
        values['cls_pred'].squeeze(-1)[mask], values['offset_pred'][mask],
        teacher_logits, teacher_offsets, args.temperature)
    loss = sum(supervised.values()) + args.score_distill_weight * kd_score + args.offset_distill_weight * kd_offset
    if not torch.isfinite(loss):
        raise RuntimeError('Non-finite training loss')
    loss.backward()
    if any(parameter.grad is not None for parameter in teacher.parameters()):
        raise RuntimeError('Frozen teacher received gradients')
    if args.max_batches:
        for name, stage in (('corners', model.keypoint_det_net),
                            ('refinement', model.cluster_refine_net), ('edges', model.edge_att_net)):
            gradients = [p.grad for p in stage.parameters() if p.grad is not None]
            if not gradients or not all(torch.isfinite(g).all() for g in gradients) or not any(g.abs().sum() > 0 for g in gradients):
                raise RuntimeError(f'Smoke test requires finite nonzero gradients in {name}')
    torch.nn.utils.clip_grad_norm_(model.parameters(), 10, error_if_nonfinite=True)
    optimizer.step()
    values = {**supervised, 'kd_score': kd_score, 'kd_offset': kd_offset, 'total': loss}
    return {key: float(value.detach()) for key, value in values.items()}, size, prediction is None


def main():
    global STOP_REQUESTED
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--initial-model', type=Path, default=PROJECT / 'models/koor-v1')
    parser.add_argument('--synthetic-dir', type=Path, default=PROJECT / 'data/prepared')
    parser.add_argument('--real-dir', type=Path, default=PROJECT / 'data/building3d/koor-v2')
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--corner-lr', type=float, default=1e-5)
    parser.add_argument('--head-lr', type=float, default=1e-4)
    parser.add_argument('--score-distill-weight', type=float, default=1.0)
    parser.add_argument('--offset-distill-weight', type=float, default=1.0)
    parser.add_argument('--temperature', type=float, default=2.0)
    parser.add_argument('--min-synthetic-retention', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max-batches', type=int, default=0)
    parser.add_argument('--checkpoint-every-batches', type=int, default=100)
    parser.add_argument('--stop-after-batches', type=int, default=0,
                        help='Controlled pause after this many optimizer steps in this invocation; for resume testing')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    numeric = (args.corner_lr, args.head_lr, args.score_distill_weight, args.offset_distill_weight,
               args.temperature, args.min_synthetic_retention)
    if (not all(math.isfinite(x) for x in numeric) or args.epochs < 1 or args.batch_size < 2 or args.batch_size % 2
            or min(args.workers, args.max_batches, args.stop_after_batches) < 0
            or args.checkpoint_every_batches < 1 or min(args.corner_lr, args.head_lr, args.temperature) <= 0
            or min(args.score_distill_weight, args.offset_distill_weight) < 0
            or args.score_distill_weight + args.offset_distill_weight <= 0
            or not 0 < args.min_synthetic_retention <= 1):
        parser.error('Invalid settings; batch size must be even and all weights finite')
    for name in ('initial_model', 'synthetic_dir', 'real_dir', 'run_dir'):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    if args.resume:
        if not (args.run_dir / 'last.pt').is_file():
            raise FileNotFoundError(args.run_dir / 'last.pt')
    elif args.run_dir.exists() and any(args.run_dir.iterdir()):
        raise FileExistsError('Use a fresh run directory or --resume')
    signal.signal(signal.SIGUSR1, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    counts, fingerprint = inspect_data(args.synthetic_dir, args.real_dir)
    config, preprocessing, metadata, weights, initial_hash = initial_release(args.initial_model)
    implementation = implementation_identity()
    settings = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    if not torch.cuda.is_available():
        raise RuntimeError('Run this trainer inside an allocated GPU batch job')
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.cuda.set_device(0)
    model = RoofNet(copy.deepcopy(config.MODEL))
    replace_batchnorm(model.cluster_refine_net)
    replace_batchnorm(model.edge_att_net)
    model.load_state_dict(weights, strict=True)
    model.cuda()
    teacher = copy.deepcopy(model.keypoint_det_net).requires_grad_(False).eval()
    teacher_hash = tensor_state_digest(teacher)
    optimizer = torch.optim.Adam([
        {'params': model.keypoint_det_net.parameters(), 'lr': args.corner_lr},
        {'params': list(model.cluster_refine_net.parameters()) + list(model.edge_att_net.parameters()), 'lr': args.head_lr},
    ], weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    cursor, history, baseline, reports = fresh_cursor(), [], None, None
    best_joint, best_joint_epoch, best_real, best_retained_epoch = -1., 0, -1., None

    if args.resume:
        saved = torch.load(args.run_dir / 'last.pt', map_location='cpu', weights_only=False)
        if saved.get('model_version') != 'koor-alex-mixed-distill-v1':
            raise ValueError('Checkpoint is not from this Alex trainer')
        if (saved['dataset_fingerprint'] != fingerprint or saved['initialization_release_sha256'] != initial_hash
                or saved['teacher_state_sha256'] != teacher_hash or saved['implementation'] != implementation
                or saved['torch_version'] != str(torch.__version__) or saved['cuda_version'] != torch.version.cuda):
            raise ValueError('Data, initialization, implementation or runtime changed since checkpoint')
        for key in ('batch_size','workers','corner_lr','head_lr','score_distill_weight',
                    'offset_distill_weight','temperature','min_synthetic_retention','seed','max_batches'):
            if saved['settings'][key] != settings[key]:
                raise ValueError(f'Resume requires original {key}')
        model.load_state_dict(saved['model'], strict=True)
        optimizer.load_state_dict(saved['optimizer'])
        scheduler.load_state_dict(saved['scheduler'])
        cursor, history = saved['cursor'], saved['history']
        baseline, reports = saved['baseline_validation'], saved['current_validation']
        best_joint, best_joint_epoch = saved['best_joint_score'], saved['best_joint_epoch']
        best_real, best_retained_epoch = saved['best_retained_real_score'], saved['best_retained_epoch']
        torch.set_rng_state(saved['torch_rng'])
        torch.cuda.set_rng_state(saved['cuda_rng'])
        np.random.set_state(saved['numpy_rng'])
        random.setstate(saved['python_rng'])
        print(f"RESUME: completed epochs={cursor['epoch']}; next batch={cursor['next_batch']}", flush=True)

    # Only development clouds/labels are opened. inspect_data reads held-out LISTS only.
    synthetic = GraphDataset(args.synthetic_dir / 'train.txt', True, args.seed)
    real = GraphDataset(args.real_dir / 'train.txt', True, args.seed)
    validation = {'synthetic': GraphDataset(args.synthetic_dir / 'val.txt', False, args.seed),
                  'real': GraphDataset(args.real_dir / 'val.txt', False, args.seed)}
    args.run_dir.mkdir(parents=True, exist_ok=True)
    if args.resume:
        # The checkpoint is authoritative if a prior process stopped between
        # replacing last.pt and writing the human-readable history file.
        write_json(args.run_dir / 'history.json', history)
    print('GPU:', torch.cuda.get_device_name(0), '| FP32', flush=True)
    print('Initialization: HF v1 epoch', metadata['selected_full_roof_epoch'], '| strict loading passed', flush=True)
    print('Split counts:', counts, flush=True)
    print(f'Each batch: {args.batch_size//2} synthetic + {args.batch_size//2} real; frozen v1 corner teacher', flush=True)
    print('Test/holdout clouds and labels are not loaded.', flush=True)
    if args.max_batches:
        print('SMOKE RUN: limited training and validation; not benchmark results', flush=True)

    def state(reason):
        return {
            'stage':'full_roof', 'model_version':'koor-alex-mixed-distill-v1',
            'run_mode':'smoke' if args.max_batches else 'full', 'epoch':cursor['epoch'],
            'cursor':copy.deepcopy(cursor), 'checkpoint_reason':reason,
            'model':model.state_dict(), 'optimizer':optimizer.state_dict(), 'scheduler':scheduler.state_dict(),
            'model_config':config, 'preprocessing':preprocessing, 'settings':settings,
            'initialization_release_sha256':initial_hash, 'initialization_v1_epoch':metadata['selected_full_roof_epoch'],
            'teacher_state_sha256':teacher_hash, 'dataset_fingerprint':fingerprint, 'split_counts':counts,
            'implementation':implementation, 'torch_version':str(torch.__version__), 'cuda_version':torch.version.cuda,
            'baseline_validation':baseline, 'current_validation':reports, 'history':history,
            'best_joint_score':best_joint, 'best_joint_epoch':best_joint_epoch,
            'best_retained_real_score':best_real, 'best_retained_epoch':best_retained_epoch,
            'torch_rng':torch.get_rng_state(), 'cuda_rng':torch.cuda.get_rng_state(),
            'numpy_rng':np.random.get_state(), 'python_rng':random.getstate(),
            'training_protocol':{'domain_ratio':[.5,.5], 'corner_distillation':'class-balanced Bernoulli KL + confidence-squared offset SmoothL1',
                                 'validation_tolerances':[.01,.02,.05], 'edge_threshold':.5,
                                 'sampling':'per-draw deterministic augmentation; shuffled domain cycles',
                                 'held_out_used_for_selection':False},
        }

    def save_last(reason):
        if tensor_state_digest(teacher) != teacher_hash:
            raise RuntimeError('Frozen teacher parameters or buffers changed')
        save_checkpoint(args.run_dir / 'last.pt', state(reason))
        print(f"CHECKPOINT: {reason}; completed epochs={cursor['epoch']}; next batch={cursor['next_batch']}", flush=True)

    def validate_both():
        result = {}
        for domain, dataset in validation.items():
            print('Validating', domain, flush=True)
            result[domain] = validate(model, dataset, args, should_stop)
        return result

    invocation_steps = 0
    try:
        if baseline is None:
            save_last('before_baseline')
            baseline = validate_both()
            reports = baseline
            print_validation('Frozen v1 validation baseline', baseline)
            best_joint = harmonic(domain_score(baseline['synthetic']), domain_score(baseline['real']))
            best_real = domain_score(baseline['real'])
            write_json(args.run_dir / 'baseline_validation.json', {'validation':baseline, 'settings':settings,
                       'dataset_fingerprint':fingerprint, 'initialization_release_sha256':initial_hash})
            save_checkpoint(args.run_dir / 'best_joint.pt', state('initialization'))
            save_last('baseline_complete')
        while cursor['epoch'] < args.epochs:
            if should_stop():
                raise InterruptedError('Stop requested before epoch')
            epoch = cursor['epoch']
            sampler = BalancedBatchSampler(len(synthetic), len(real), args.batch_size, args.seed, epoch, cursor['next_batch'])
            limit = min(args.max_batches, sampler.total_batches) if args.max_batches else sampler.total_batches
            if cursor['next_batch'] > limit:
                raise ValueError('Saved batch cursor exceeds epoch limit')
            print(f"Epoch {epoch+1}/{args.epochs}; next batch={cursor['next_batch']}; epoch batches={limit}", flush=True)
            model.train()
            teacher.eval()
            if cursor['next_batch'] < limit:
                dataset = ReplayDataset(synthetic, real, args.seed, epoch)
                loader = DataLoader(dataset, batch_sampler=sampler, num_workers=args.workers,
                                    collate_fn=tagged_collate, pin_memory=True, worker_init_fn=worker_seed,
                                    generator=torch.Generator().manual_seed(args.seed+epoch))
                for batch in loader:
                    if should_stop():
                        raise InterruptedError('Signal requested stop')
                    step = cursor['next_batch']
                    if step >= limit:
                        break
                    losses, count, empty = training_step(model, teacher, batch, optimizer, args)
                    cursor['next_batch'] += 1
                    cursor['samples'] += count
                    cursor['empty_batches'] += empty
                    for key, value in losses.items():
                        cursor['sums'][key] += value * count
                    invocation_steps += 1
                    if step % 100 == 0 or args.max_batches:
                        print(f'  batch {step}: ' + ', '.join(f'{k}={v:.5f}' for k,v in losses.items()), flush=True)
                    if args.stop_after_batches and invocation_steps >= args.stop_after_batches:
                        STOP_REQUESTED = True
                    if should_stop():
                        raise InterruptedError('Requested safe pause after batch')
                    if cursor['next_batch'] % args.checkpoint_every_batches == 0:
                        save_last('periodic_batch')
                    if cursor['next_batch'] >= limit:
                        break
                del loader
            training = train_report(cursor)
            save_last('before_validation')
            reports = validate_both()
            if should_stop():
                raise InterruptedError('Stop requested after validation')
            scheduler.step()
            joint = harmonic(domain_score(reports['synthetic']), domain_score(reports['real']))
            status = selection_status(reports, baseline, args.min_synthetic_retention)
            better_joint = joint > best_joint
            better_retained = status['eligible'] and domain_score(reports['real']) > best_real
            if better_joint:
                best_joint, best_joint_epoch = joint, epoch+1
            if better_retained:
                best_real, best_retained_epoch = domain_score(reports['real']), epoch+1
            history.append({'epoch':epoch+1,'train':training,'validation':reports,'joint_score':joint,'retention_status':status})
            cursor = fresh_cursor(epoch+1)
            if better_joint:
                save_checkpoint(args.run_dir / 'best_joint.pt', state('joint'))
            if better_retained:
                save_checkpoint(args.run_dir / 'best_retained.pt', state('retention_qualified'))
            save_last('epoch_complete')
            write_json(args.run_dir / 'history.json', history)
            print('Training:', training, flush=True)
            print_validation('Validation', reports)
            print(f'Joint score={joint:.4f}; best={best_joint:.4f} at epoch {best_joint_epoch}', flush=True)
            print('Retention gate:', status, flush=True)
            print('PASS: frozen teacher parameters and buffers unchanged', flush=True)
        print('COMPLETED:', args.run_dir, flush=True)
        if best_retained_epoch is None:
            print('No epoch met retention/improvement requirements; best_retained.pt was not created.', flush=True)
    except InterruptedError as error:
        save_last('safe_pause')
        print('PAUSED SAFELY:', error, flush=True)
        print('Resume the same run with --resume. No optimizer updates will be replayed.', flush=True)


if __name__ == '__main__':
    main()
