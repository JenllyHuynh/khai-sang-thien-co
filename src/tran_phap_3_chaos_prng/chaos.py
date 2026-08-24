import numpy as np

SIGMA = 10.0
RHO = 28.0
BETA = 8.0 / 3.0

def _lorenz_deriv(state: np.ndarray, sigma: float, rho: float, beta: float) -> np.ndarray:
    x, y, z = state
    return np.array([
        sigma * (y - x),
        x * (rho - z) - y,
        x * y - beta * z,
    ])

# Tích phân hệ Lorenz bằng Runge-Kutta bậc 4 (RK4)
def simulate_lorenz(
    initial_state, n_steps: int = 6000, dt: float = 0.01,
    sigma: float = SIGMA, rho: float = RHO, beta: float = BETA,
) -> np.ndarray:
    # Trả về quỹ đạo dạng mảng (n_steps + 1, 3) - mỗi hàng là (x, y, z) tại một bước thời gian
    trajectory = np.zeros((n_steps + 1, 3), dtype=np.float64)
    state = np.array(initial_state, dtype=np.float64)
    trajectory[0] = state

    for i in range(n_steps):
        k1 = _lorenz_deriv(state, sigma, rho, beta)
        k2 = _lorenz_deriv(state + dt / 2 * k1, sigma, rho, beta)
        k3 = _lorenz_deriv(state + dt / 2 * k2, sigma, rho, beta)
        k4 = _lorenz_deriv(state + dt * k3, sigma, rho, beta)
        state = state + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        trajectory[i + 1] = state

    return trajectory

# Chạy 2 quỹ đạo với điều kiện ban đầu sai khác cực nhỏ (perturbation), trả về khoảng cách Euclid giữa 2 quỹ đạo theo thời gian
def butterfly_effect(
    initial_state, perturbation: float = 1e-8, n_steps: int = 3000, dt: float = 0.01,
) -> tuple:
    state_a = np.array(initial_state, dtype=np.float64)
    state_b = state_a + np.array([perturbation, 0.0, 0.0])

    traj_a = simulate_lorenz(state_a, n_steps=n_steps, dt=dt)
    traj_b = simulate_lorenz(state_b, n_steps=n_steps, dt=dt)

    distance = np.linalg.norm(traj_a - traj_b, axis=1)
    times = np.arange(n_steps + 1) * dt
    return times, distance
