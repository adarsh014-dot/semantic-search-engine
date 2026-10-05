from backend.evaluation import metrics


def test_metrics_basic():
    flags = [False, True, False, True]
    assert metrics.precision_at_k(flags, 4) == 0.5
    assert metrics.recall_at_k(flags, 1) == 0.0 and metrics.recall_at_k(flags, 2) == 1.0
    assert metrics.reciprocal_rank(flags) == 0.5
    assert metrics.reciprocal_rank([False]) == 0.0
    assert metrics.hit_rate([[True], [False]], 1) == 0.5
