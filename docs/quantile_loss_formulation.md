# Mathematical Foundations: Quantile & Pinball Loss

The Quantile Loss (Pinball Loss) $\mathcal{L}_{q}(y, \hat{y})$ for a target quantile $q \in (0, 1)$ is defined as:

$$\mathcal{L}_{q}(y, \hat{y}) = \max \left( q(y - \hat{y}), (q - 1)(y - \hat{y}) \right)$$

Continuous Ranked Probability Score (CRPS) is computed across quantiles $q \in \{0.1, 0.5, 0.9\}$:

$$\text{CRPS} = \frac{1}{K} \sum_{k=1}^K \mathcal{L}_{q_k}(y, \hat{y}^{(q_k)})$$
