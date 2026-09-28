# Metropolis-Hastings MCMC Sampler Skill

Markov Chain Monte Carlo (MCMC) sampler for Bayesian posterior estimation and high-dimensional probabilistic distribution exploration.

```mermaid
flowchart TD
    Current["Current State x_t"] --> Propose["Propose New Candidate x' ~ Q(x' | x_t)"]
    Propose --> Acceptance["Acceptance Ratio α = min(1, π(x') / π(x_t))"]
    Acceptance --> Decision{"Uniform Random u < α?"}
    Decision -- Yes --> Accept["Accept: x_{t+1} = x'"]
    Decision -- No --> Reject["Reject: x_{t+1} = x_t"]
    Accept --> Next["Stationary Distribution Trace"]
    Reject --> Next
```

## Features
- **100% Python Standard Library**: Unnormalized log-probability evaluation.
- **Detailed Balance Guarantee**: Provable asymptotic convergence to true target density.
