# V10: test visual routing to the adapted policy

All images came from simulation. No controller test used these gates.

We tried three small visual classifiers. Each gate had a fixed validation limit of 5% false positives. We kept the gate only if it met that limit on fresh images.

| Gate | Fresh tipped recall | Fresh upright false-positive rate |
|---|---:|---:|
| Linear frozen-feature gate | 5.0% | 2.5% |
| 32-unit MLP gate | 2.5% | 2.5% |
| 32-pixel crop classifier | 2.5% | 12.5% |

The first two gates missed almost every tipped object. The crop gate also exceeded the false-positive limit. We did not route the policy in controller tests. The reports and fixed thresholds are in `visual-router/`, `../world-v10-mlp/visual-router/`, and `../world-v10-crop/router/`.
