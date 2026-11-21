from recsys.eval.slices import evaluate_cohorts


def test_perfect_cohort_scores_one():
    cases = [("premium", ["a", "b"], {"a"}), ("premium", ["c", "d"], {"c"})]
    out = evaluate_cohorts(cases, k=2, min_support=1)
    assert out["micro_precision_at_k"] == 0.5
    assert out["slices"]["premium"].hit_rate == 1.0
    assert out["slices"]["premium"].precision_at_k == 0.5


def test_missed_recommendation_scores_zero():
    cases = [("free", ["x", "y"], {"a"})]
    out = evaluate_cohorts(cases, k=2, min_support=1)
    assert out["slices"]["free"].hit_rate == 0.0
    assert out["slices"]["free"].mrr == 0.0


def test_cohorts_sorted_and_low_support_flagged():
    big = [("premium", ["a"], {"a"})] * 3
    small = [("free", ["b"], {"b"})]
    out = evaluate_cohorts(big + small, k=1, min_support=3)
    assert list(out["slices"]) == ["free", "premium"]
    assert out["low_support_slices"] == ["free"]
    assert out["macro_precision_at_k"] == 1.0


def test_k_limits_precision():
    cases = [("premium", ["a", "b", "c"], {"c"})]
    out = evaluate_cohorts(cases, k=2, min_support=1)
    assert out["slices"]["premium"].precision_at_k == 0.0
    out2 = evaluate_cohorts(cases, k=3, min_support=1)
    assert out2["slices"]["premium"].precision_at_k == round(1 / 3, 4)


def test_invalid_inputs_rejected():
    case = [("premium", ["a"], {"a"})]
    for kwargs in ({"cases": [], "k": 3, "min_support": 1},
                   {"cases": case, "k": 0, "min_support": 1},
                   {"cases": case, "k": 3, "min_support": 0}):
        cases = kwargs.pop("cases")
        try:
            evaluate_cohorts(cases, **kwargs)
        except ValueError:
            continue
        raise AssertionError(f"accepted invalid input: {kwargs}")
