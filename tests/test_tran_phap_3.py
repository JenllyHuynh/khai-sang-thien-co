import numpy as np

from src.tran_phap_3_chaos_prng.chaos import butterfly_effect, simulate_lorenz
from src.tran_phap_3_chaos_prng.prng_demo import (
    WeakLCG,
    attempt_state_reconstruction_crack,
    generate_strong_csprng,
    generate_weak_lcg,
)
from src.tran_phap_3_chaos_prng.simulate import run_simulation
from src.tran_phap_3_chaos_prng.stats import estimate_lyapunov_slope

def test_simulate_lorenz_shape_and_determinism():
    traj1 = simulate_lorenz((1.0, 1.0, 1.0), n_steps=500, dt=0.01)
    traj2 = simulate_lorenz((1.0, 1.0, 1.0), n_steps=500, dt=0.01)
    assert traj1.shape == (501, 3)
    # Cùng điều kiện ban đầu -> quỹ đạo phải HOÀN TOÀN giống nhau (tất định)
    assert np.allclose(traj1, traj2)

def test_butterfly_effect_diverges():
    times, distance = butterfly_effect((1.0, 1.0, 1.0), perturbation=1e-8, n_steps=3000, dt=0.01)
    assert times.shape == distance.shape == (3001,)
    assert distance[0] > 0  # có nhiễu ban đầu
    # Khoảng cách cuối cùng phải lớn hơn RẤT NHIỀU so với nhiễu ban đầu (1e-8)
    assert distance[-1] > distance[0] * 1000

def test_estimate_lyapunov_slope_positive():
    times, distance = butterfly_effect((1.0, 1.0, 1.0), perturbation=1e-8, n_steps=3000, dt=0.01)
    result = estimate_lyapunov_slope(times, distance)
    assert result["slope"] is not None
    assert result["slope"] > 0  # phân kỳ theo hàm mũ -> slope dương
    assert result["doubling_time"] is not None

def test_generate_prng_shapes_and_range():
    weak = generate_weak_lcg(1000, seed=1)
    strong = generate_strong_csprng(1000, seed=1)
    for arr in (weak, strong):
        assert arr.shape == (1000,)
        assert arr.min() >= 0.0 and arr.max() < 1.0

def test_crack_weak_lcg_succeeds():
    weak = generate_weak_lcg(500, seed=7)
    result = attempt_state_reconstruction_crack(weak)
    assert result["cracked"] is True
    assert result["max_abs_error"] < 1e-9


def test_crack_attempt_fails_on_strong_csprng():
    strong = generate_strong_csprng(500, seed=7)
    result = attempt_state_reconstruction_crack(strong)
    assert result["cracked"] is False
    assert result["max_abs_error"] > 0.01

def test_weak_lcg_reproducible_with_same_seed():
    a = generate_weak_lcg(100, seed=42)
    b = generate_weak_lcg(100, seed=42)
    assert np.array_equal(a, b)

def test_run_simulation_end_to_end():
    result = run_simulation(
        n_steps_attractor=500, dt=0.01, n_steps_butterfly=500, n_prng_samples=500, seed=1
    )
    assert result["trajectory"].shape == (501, 3)
    assert result["crack_weak_result"]["cracked"] is True
    assert result["naive_crack_on_strong"]["cracked"] is False
