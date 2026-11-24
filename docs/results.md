# Reading Koor v1's results

I wanted to test two things: whether the model could reconstruct synthetic roofs it hadn't seen during training, and whether those skills carried over to real LiDAR. The second test turned out to be much harder.

The results below use the best full-model checkpoint, selected by validation at epoch 29. Predictions were made from point clouds alone, with target corners and edges loaded afterward for scoring. The edge threshold stayed at 0.5 for both tests.

| Test | Buildings | Corner tolerance | Corner micro-F1 | Edge micro-F1 |
|---|---:|---:|---:|---:|
| Point2Roof synthetic test | 1,600 | 1% | 0.5874 | 0.4079 |
| Point2Roof synthetic test | 1,600 | 2% | 0.8891 | 0.8372 |
| Point2Roof synthetic test | 1,600 | 5% | 0.9549 | 0.9445 |
| Building3D real-data holdout | 561 | 1% | 0.0032 | 0.0000 |
| Building3D real-data holdout | 561 | 2% | 0.0212 | 0.0028 |
| Building3D real-data holdout | 561 | 5% | 0.1770 | 0.0482 |

## What counts as a correct prediction?

A roof point cloud is centered and divided by its maximum bounding-box span, using the same scale for all three axes. A 5% corner tolerance allows an error of up to 0.05 in these normalized coordinates. For a building spanning 20 source-coordinate units, that would allow an error of 1 source unit. The test does not assume those units are metres.

Predicted corners are paired with true corners using minimum-total-distance Hungarian assignment. A pair counts as a match only if it passes the distance threshold. Each true corner can match at most one prediction.

For an edge to count, both predicted endpoints must match the endpoints of a true edge. Direction doesn't matter. A line connecting the wrong corners is still an incorrect edge, even if both corners are individually well placed.

## What F1 tells us

Precision asks how many predictions were correct. Recall asks how much of the true roof was recovered. F1 balances those two measures.

The table uses **micro-F1**: counts are pooled across all buildings before calculating the score. The saved reports also contain **macro-F1**, which calculates F1 per building and averages those scores, giving each building equal weight.

Neither measure is the percentage of perfectly reconstructed buildings. A roof can contribute many correct corners while still having a missing ridge or an extra connection. The reported corner-position error also leaves out unmatched corners, so it needs to be read alongside recall.

## Where the model struggled

Even on synthetic data, accuracy falls at the 1% tolerance. That shows the high 5% scores don't tell the whole story about corner placement.

The real-data scores are much lower. Differences in shape, sampling, noise, or preprocessing could contribute to that gap; this evaluation doesn't establish which factor is responsible. All 561 real buildings produced corner proposals, but 37 had no accepted edges. Producing something for a building is a much weaker claim than reconstructing it correctly.

## Which data was held out?

The synthetic experiment used 14,400 training, 1,599 validation, and 1,600 test samples. Matching normalized vertex layouts were grouped across splits; one source training sample matching test geometry was excluded. Rotated copies and other near-duplicates may still escape that check.

The Building3D test uses a custom, spatially separated holdout from its labelled Entry-level data. Koor v1 was not trained on it. The official Entry-level test clouds have no wireframe labels in the downloaded archive, so they weren't used for numeric scoring.

These are results from the Koor evaluation procedure. Different splits, matching rules, or tolerances can change the numbers, so I don't claim a direct comparison with published benchmark tables.

## Files behind the table

- [Synthetic test report](../v1/evaluation/synthetic_test.json)
- [Building3D report](../v1/evaluation/building3d_zero_shot.json)
- [Checkpoint selection and training settings](../v1/training_metadata.json)
- [Target-free evaluation code](../v1/scripts/evaluate_roofs.py)
- [Clean-environment release verification](../v1/provenance/release_verification.json)

The last file records a packaging and inference check on B200. Its demo is a small mathematical roof, not another accuracy benchmark.

## V2: mixed training on synthetic and Building3D roofs

V2 was trained on Alex using synthetic replay and a frozen v1 corner teacher. Its epoch-14 checkpoint was selected on the joint validation score before held-out evaluation. The [v2 reports](../v2/evaluation/) rerun v1 and v2 together, using the same 1,600 synthetic test roofs and 561-building spatial holdout. These reports are a separate Alex measurement from the original v1 B200 reports above; use the paired reports for v1/v2 changes.

At 5% tolerance, synthetic vertex/edge F1 changes from 0.9547/0.9441 to 0.9451/0.9276. Building3D changes from 0.1763/0.0482 to 0.5275/0.3979. All six real-data F1 scores improve, and all six synthetic F1 scores decline. At 1%, synthetic edge F1 falls from 0.4076 to 0.3334. Building3D roofs with no accepted edges increase from 37 to 45; both models have zero empty edge predictions on the synthetic test.

The experiment therefore shows adaptation with a retention tradeoff. It did not pass the original requirement to retain all synthetic validation metrics while improving the real score. `best_retained.pt` was never created; the reported model is the validation-selected `best_joint.pt`. The [full v2 guide](../v2/README.md) includes every tolerance and training details.

The custom spatial split, normalized tolerance and graph-matching protocol remain the same types of Koor evaluation described above. Higher aggregate F1 does not establish valid closed roof faces, complete buildings or reliable individual predictions. None of these tests establish performance from photographs, map coordinates or addresses.
