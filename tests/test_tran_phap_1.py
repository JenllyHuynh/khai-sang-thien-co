import numpy as np

from src.tran_phap_1_monte_carlo.encode import decode_ticket, encode_tickets
from src.tran_phap_1_monte_carlo.lottery_config import N_NUMBERS, PICK
from src.tran_phap_1_monte_carlo.ticket_generator import generate_bias, generate_uniform


def test_generate_uniform_shape_and_validity():
    rng = np.random.default_rng(0)
    tickets = generate_uniform(1000, rng)
    assert tickets.shape == (1000, PICK)
    assert tickets.min() >= 1 and tickets.max() <= N_NUMBERS
    # Không được trùng số trong cùng 1 vé
    for row in tickets[:100]:
        assert len(set(row.tolist())) == PICK


def test_generate_bias_shape_and_validity():
    rng = np.random.default_rng(0)
    tickets = generate_bias(1000, rng)
    assert tickets.shape == (1000, PICK)
    assert tickets.min() >= 1 and tickets.max() <= N_NUMBERS
    for row in tickets[:100]:
        assert len(set(row.tolist())) == PICK


def test_encode_decode_roundtrip():
    rng = np.random.default_rng(1)
    tickets = generate_uniform(200, rng)
    ids = encode_tickets(tickets)
    for i in range(20):
        decoded = decode_ticket(ids[i])
        assert decoded == tuple(sorted(tickets[i].tolist()))


def test_bias_group_clusters_more_than_uniform():
    """Chân lý cốt lõi: Phe Tình Cảm (thiên vị) phải có tỷ lệ trùng vé
    cao hơn hẳn Phe Vô Vi (random thuần) với cùng số lượng người chơi."""
    rng = np.random.default_rng(2)
    n = 30_000
    uniform_tickets = generate_uniform(n, rng)
    bias_tickets = generate_bias(n, rng)

    uniform_ids = encode_tickets(uniform_tickets)
    bias_ids = encode_tickets(bias_tickets)

    n_unique_uniform = np.unique(uniform_ids).shape[0]
    n_unique_bias = np.unique(bias_ids).shape[0]

    # Phe thiên vị dồn vào vùng số hẹp hơn -> ít tổ hợp độc nhất hơn
    assert n_unique_bias < n_unique_uniform
