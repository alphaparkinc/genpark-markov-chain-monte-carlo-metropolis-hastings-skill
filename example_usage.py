"""Example demonstrating MCMC sampling from standard normal."""
from client import MetropolisHastingsSampler

def main():
    target = lambda x: -0.5 * (x**2)
    stats = MetropolisHastingsSampler.sample_1d(target, x0=0.0, num_samples=1500)
    print("MCMC Sampler Statistics:")
    print("  Acceptance Rate:", stats["acceptance_rate"])
    print("  Empirical Mean:", stats["sample_mean"])
    print("  Empirical Variance:", stats["sample_variance"])

if __name__ == "__main__":
    main()
