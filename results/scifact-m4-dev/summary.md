# SciFact control run: m4-dev

dev split, 339 (claim, abstract) pairs, 5 repeats per backend; release sha256 11c621288d41. Rates are given with 1 sigma Clopper-Pearson intervals.

## Accuracy and calibration (3-option verdict, every pair, stated mean)

| Backend | Accuracy | Single-call accuracy | Top-label ECE | ECE floor | Classwise ECE | Brier | Cost |
|---|---|---|---|---|---|---|---|
| typesafe/jev-1.14-20261014 | 85.5% (+1.9 / -2.2) | 85.0% (+0.9 / -0.9) | 0.064 | 0.027 (95th pct 0.042) | 0.057 | 0.226 | $0.056 |
| anthropic/claude-haiku-5.5/effort=medium | 82.9% (+2.1 / -2.3) | 82.8% (+0.9 / -1.0) | 0.051 | 0.034 (95th pct 0.053) | 0.038 | 0.265 | $0.323 |
| majority class (supports) | 40.7% (+2.8 / -2.8) | | | | | | |

ECE floor: the top-label ECE a perfectly calibrated model stating the same probabilities shows from sampling noise alone (4000 simulated draws; mean and 95th percentile).

## Working points (reporting half, top stated-mean probability)

Split: selection {'supports': 69, 'not_enough_info': 65, 'contradicts': 35}, reporting {'not_enough_info': 65, 'supports': 69, 'contradicts': 36}. An answer scores +1 if right and -c if wrong; an escalated item 0. Each cell is threshold | coverage | accuracy of the answered | utility per item. The oracle row is a biased upper bound.

| Backend | c | Row | Threshold | Coverage | Accuracy answered | Utility |
|---|---|---|---|---|---|---|
| jev | 1 | frozen | 0.560 | 94% | 88.8% | +0.729 |
| jev | 1 | stated | 0.500 | 98% | 88.0% | +0.741 |
| jev | 1 | oracle | 0.460 | 99% | 87.6% | +0.747 |
| jev | 3 | frozen | 0.846 | 82% | 92.8% | +0.582 |
| jev | 3 | stated | 0.750 | 88% | 91.9% | +0.594 |
| jev | 3 | oracle | 0.874 | 79% | 94.1% | +0.606 |
| jev | 9 | frozen | 0.948 | 71% | 95.0% | +0.353 |
| jev | 9 | stated | 0.900 | 75% | 94.5% | +0.341 |
| jev | 9 | oracle | 0.992 | 44% | 100.0% | +0.435 |
| haiku | 1 | frozen | 0.626 | 94% | 86.8% | +0.688 |
| haiku | 1 | stated | 0.500 | 99% | 84.0% | +0.676 |
| haiku | 1 | oracle | 0.668 | 92% | 88.5% | +0.706 |
| haiku | 3 | frozen | 0.654 | 92% | 87.9% | +0.476 |
| haiku | 3 | stated | 0.750 | 87% | 89.9% | +0.518 |
| haiku | 3 | oracle | 0.760 | 85% | 90.3% | +0.524 |
| haiku | 9 | frozen | 0.944 | 15% | 96.0% | +0.088 |
| haiku | 9 | stated | 0.900 | 61% | 91.3% | +0.076 |
| haiku | 9 | oracle | 0.946 | 14% | 100.0% | +0.135 |

## Agreement (jev rows, haiku columns)

| | supports | contradicts | not_enough_info |
|---|---|---|---|
| supports | 128 | 1 | 8 |
| contradicts | 13 | 65 | 6 |
| not_enough_info | 11 | 2 | 105 |

Same answer on 87.9% of pairs.

## Jev's confidence vs (K * pmax - 1) / (K - 1)

| Question | K | Calls | Max abs residual | Mean abs residual |
|---|---|---|---|---|
| scifact.verdict | 3 | 1695 | 0.0050 | 0.0016 |
| scifact.supports | 2 | 500 | 0.0000 | 0.0000 |
