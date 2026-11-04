### Judge vs reference labels (n=53)
- exact agreement 0.981, Cohen's kappa (3 classes) 0.972
- binary (fully correct or not): agreement 1.000, kappa 1.000
- 95% Wilson interval on exact agreement: 0.901 to 0.997
- confusion, label to judge score: {"0": {"0": 15, "1": 1}, "1": {"1": 17}, "2": {"2": 20}}
- generated rows only (n=40): agreement 1.000, kappa 3-class 1.000, binary kappa 1.000
- handwritten rows only (n=13): agreement 0.923, kappa 3-class 0.885, binary kappa 1.000

### Position bias (17 good/bad pairs, each judged in both orders)
| | pairs | picks first | right when good is first | right when good is second | verdict flips on swap |
|---|---:|---:|---:|---:|---:|
| plain | 17 | 0.500 | 1.000 | 1.000 | 0.000 |
| bad answer padded | 17 | 0.500 | 1.000 | 1.000 | 0.000 |

### Length bias (n=53, each answer judged again with about 45 words of filler appended)
- mean score change +0.019, raised 0.019, lowered 0.000, wrong or partial answers raised 0.030
- Spearman(answer length, judge score) 0.546; within a label: label 0: 0.029, label 1: n/a, label 2: n/a
