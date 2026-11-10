"""
Trận Pháp 1: Vạn Kiếp Quy Tông (Monte Carlo Simulator)

"Chọn số đẹp không làm tăng Thiên Mệnh, nhưng sẽ khiến tiền thưởng
bị phân tán như cát bụi."

Chạy nhanh:
    python -m src.tran_phap_1_monte_carlo.run --n-players 1000000
"""
from .lottery_config import N_NUMBERS, PICK, BiasConfig
from .simulate import run_simulation
from .stats import cluster_stats, top_popular_combos
from .encode import encode_tickets, decode_ticket

__all__ = [
    "N_NUMBERS",
    "PICK",
    "BiasConfig",
    "run_simulation",
    "cluster_stats",
    "top_popular_combos",
    "encode_tickets",
    "decode_ticket",
]
