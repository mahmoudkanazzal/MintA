# MintA on PDNS-Net

Feature attack and combined feature+edge attack on
the public PDNS-Net graph. One linearized-GCN surrogate trained only on the target detector's
answers for queried domains; the detector is never queried during scoring. ASR and NFR are
mean with 95% confidence interval over trials. Pre-attack detection rate is 97.3%.

| experiment | perturbed_domains | ASR | NFR | detected_after | trials |
|---|---|---|---|---|---|
| feature | 1 | 0.009 [0.007, 0.010] | 0.000 [0.000, 0.000] | 96.4 | 27 |
| feature | 5 | 0.046 [0.043, 0.050] | 0.000 [0.000, 0.000] | 92.8 | 27 |
| feature | 20 | 0.177 [0.168, 0.186] | 0.000 [0.000, 0.000] | 80.0 | 27 |
| feature | 40 | 0.342 [0.321, 0.363] | 0.000 [0.000, 0.000] | 64.0 | 27 |
| feature | 60 | 0.482 [0.453, 0.510] | 0.000 [0.000, 0.000] | 50.4 | 27 |
| feature | 80 | 0.617 [0.580, 0.653] | 0.000 [0.000, 0.000] | 37.3 | 27 |
| feature | 100 | 0.718 [0.682, 0.755] | 0.000 [0.000, 0.000] | 27.4 | 27 |
| joint | 1 | 0.009 [0.007, 0.011] | 0.000 [0.000, 0.000] | 96.4 | 27 |
| joint | 5 | 0.048 [0.042, 0.054] | 0.000 [0.000, 0.000] | 92.6 | 27 |
| joint | 20 | 0.194 [0.181, 0.206] | 0.016 [-0.017, 0.049] | 78.5 | 27 |
| joint | 40 | 0.368 [0.337, 0.400] | 0.000 [0.000, 0.000] | 61.4 | 27 |
| joint | 60 | 0.523 [0.481, 0.564] | 0.000 [0.000, 0.000] | 46.4 | 27 |
| joint | 80 | 0.679 [0.629, 0.728] | 0.000 [0.000, 0.000] | 31.3 | 27 |
| joint | 100 | 0.851 [0.811, 0.891] | 0.000 [0.000, 0.000] | 14.5 | 27 |

Headline (100 perturbed domains): feature ASR 0.718, combined ASR 0.851.
Run: 27 of 30 trials; 3 trials excluded by a GPU out-of-memory failure during scoring, which is a
resource limit uncorrelated with attack outcome. Numbers come from the released code's own
`results_table.md`.
