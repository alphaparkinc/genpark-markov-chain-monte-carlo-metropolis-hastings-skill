"""Metropolis-Hastings Markov Chain Monte Carlo (MCMC) Sampler.
100% Python Standard Library.
"""

import math
import random

class MetropolisHastingsSampler:
    """Samples from arbitrary unnormalized target probability distributions p(x)."""
    @staticmethod
    def sample_1d(target_log_prob, x0=0.0, proposal_std=1.0, num_samples=1000, seed=42):
        rng = random.Random(seed)
        samples = []
        current_x = x0
        current_log_p = target_log_prob(current_x)
        accepted = 0
        
        for _ in range(num_samples):
            proposal = rng.gauss(current_x, proposal_std)
            prop_log_p = target_log_prob(proposal)
            
            log_alpha = prop_log_p - current_log_p
            if math.log(rng.random()) < log_alpha:
                current_x = proposal
                current_log_p = prop_log_p
                accepted += 1
                
            samples.append(round(current_x, 5))
            
        mean = sum(samples) / num_samples
        variance = sum((x - mean)**2 for x in samples) / (num_samples - 1)
        return {
            "samples_count": len(samples),
            "acceptance_rate": round(accepted / num_samples, 4),
            "sample_mean": round(mean, 4),
            "sample_variance": round(variance, 4)
        }
