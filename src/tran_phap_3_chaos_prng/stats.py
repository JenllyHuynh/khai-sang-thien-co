import numpy as np

def estimate_lyapunov_slope(times: np.ndarray, distance: np.ndarray, distance_ceiling: float = 1.0) -> dict:
    mask = (distance > 0) & (distance < distance_ceiling)
    if mask.sum() < 10:
        return {"slope": None, "doubling_time": None, "n_points_used": int(mask.sum())}

    slope, _intercept = np.polyfit(times[mask], np.log(distance[mask]), 1)
    doubling_time = float(np.log(2) / slope) if slope > 0 else None
    return {
        "slope": float(slope),
        "doubling_time": doubling_time,
        "n_points_used": int(mask.sum()),
    }
