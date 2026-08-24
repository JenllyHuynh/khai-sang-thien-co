from .chaos import butterfly_effect, simulate_lorenz
from .prng_demo import attempt_state_reconstruction_crack, generate_strong_csprng, generate_weak_lcg
from .stats import estimate_lyapunov_slope

DEFAULT_INITIAL_STATE = (1.0, 1.0, 1.0)

def run_simulation(
    n_steps_attractor: int = 6000,
    dt: float = 0.01,
    n_steps_butterfly: int = 3000,
    n_prng_samples: int = 5000,
    seed: int = 42,
) -> dict:
    trajectory = simulate_lorenz(DEFAULT_INITIAL_STATE, n_steps=n_steps_attractor, dt=dt)

    times, distance = butterfly_effect(
        DEFAULT_INITIAL_STATE, perturbation=1e-8, n_steps=n_steps_butterfly, dt=dt
    )
    lyapunov = estimate_lyapunov_slope(times, distance)

    weak_samples = generate_weak_lcg(n_prng_samples, seed=seed)
    strong_samples = generate_strong_csprng(n_prng_samples, seed=seed)

    crack_weak_result = attempt_state_reconstruction_crack(weak_samples)
    naive_crack_on_strong = attempt_state_reconstruction_crack(strong_samples)

    return {
        "trajectory": trajectory,
        "butterfly_times": times,
        "butterfly_distance": distance,
        "lyapunov": lyapunov,
        "weak_lcg_samples": weak_samples,
        "strong_csprng_samples": strong_samples,
        "crack_weak_result": crack_weak_result,
        "naive_crack_on_strong": naive_crack_on_strong,
    }
