from __future__ import annotations

import numpy as np


def roc_curve_from_score(
    score: np.ndarray,
    is_laundering: np.ndarray,
    weight: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sweep every distinct value of `score` as a "flag if score >= threshold" rule and
    return (thresholds, alpha, beta), sorted from the strictest threshold (alpha=0) to the
    laxest (alpha=1).

    beta = Pr(alarm | laundering) and alpha = Pr(alarm | legitimate) -- the paper's
    true-/false-positive rates (Section 2.1.1), i.e. the detector's full ROC curve rather
    than the four ad hoc budgets in the notebook's `static` table.

    `weight` defaults to one-per-row, which makes alpha/beta the literal per-transaction
    probabilities the paper's model assumes. Pass `weight=amount` to get value-weighted
    rates instead (e.g. "share of laundered dollars caught") if that is the more useful
    operating view for your own write-up.
    """
    score = np.asarray(score, dtype=np.float64)
    is_laundering = np.asarray(is_laundering, dtype=bool)
    if weight is None:
        weight = np.ones_like(score)
    else:
        weight = np.asarray(weight, dtype=np.float64)

    order = np.argsort(-score)
    score_sorted = score[order]
    laund_sorted = is_laundering[order]
    w_sorted = weight[order]

    laund_w = np.where(laund_sorted, w_sorted, 0.0)
    legit_w = np.where(~laund_sorted, w_sorted, 0.0)

    total_laund = laund_w.sum()
    total_legit = legit_w.sum()
    if total_laund == 0 or total_legit == 0:
        raise ValueError("need at least one laundering and one legitimate row to build an ROC curve")

    cum_laund = np.cumsum(laund_w)
    cum_legit = np.cumsum(legit_w)

    # A threshold can only be set between distinct score values, so collapse ties to the
    # last row of each run: that is where "flag if score >= threshold" changes who is flagged.

    # row-wise merging
    run_ends = np.r_[np.flatnonzero(np.diff(score_sorted)), len(score_sorted) - 1]

    thresholds = score_sorted[run_ends]
    beta = cum_laund[run_ends] / total_laund
    alpha = cum_legit[run_ends] / total_legit
    return thresholds, alpha, beta


def control_a_roc(amount: np.ndarray, is_laundering: np.ndarray):
    """'Flag if Amount >= X' for every X at once -- generalizes the notebook's control_A,
    which only evaluated X at four budgets, into its full ROC curve."""
    return roc_curve_from_score(amount, is_laundering)


def control_b_roc(distinct_receiver_counts: np.ndarray, is_laundering: np.ndarray):
    """'Flag if distinct receivers in the window > K' for every K at once -- generalizes
    the notebook's control_B into its full ROC curve."""
    return roc_curve_from_score(distinct_receiver_counts, is_laundering)


if __name__ == "__main__":
    # Made-up example.
    score = np.array([50, 200, 75, 400, 120, 400, 30])
    is_laundering = np.array([False, True, False, True, False, True, False])
    # Weight each row by its own amount, per the docstring's "share of laundered
    # dollars caught" use case -- a $400 hit should count for more than a $50 one.
    weight = score

    order = np.argsort(-np.array(score))
    score_sorted = score[order]
    laund_sorted = is_laundering[order]
    w_sorted = weight[order]
    print(order)
    print(score_sorted)
    print(laund_sorted)

    laund_w = np.where(laund_sorted, w_sorted, 0.0)
    legit_w = np.where(~laund_sorted, w_sorted, 0.0)
    print(laund_w)
    print(legit_w)

    cum_laund = np.cumsum(laund_w)
    cum_legit = np.cumsum(legit_w)
    print(cum_laund)
    print(cum_legit)

    V = np.array([1, 2, 3, 4, 5, 6])
    Y = np.array([7, 8, 9, 10, 11, 12])

    mix = np.r_[V, Y]
    # print(mix)

    run_ends = np.r_[np.flatnonzero(np.diff(score_sorted)), len(score_sorted) - 1]
    print(run_ends)
    threshold = score_sorted[run_ends]
    print(threshold)

    total_laund = laund_w.sum()
    total_legit = legit_w.sum()

    beta = cum_laund[run_ends] / total_laund
    alpha = cum_legit[run_ends] / total_legit
    print(beta)
    print(alpha)




    # print(f"\n{'':>10}{'unweighted (per-transaction)':>32}{'weighted (per-dollar)':>26}")
    # print(f"{'threshold':>10}{'alpha':>16}{'beta':>16}{'alpha':>14}{'beta':>12}")
    # thresholds, alpha, beta = roc_curve_from_score(amount, is_laundering)
    # _, w_alpha, w_beta = roc_curve_from_score(amount, is_laundering, weight=weight)
    # for t, a, b, wa, wb in zip(thresholds, alpha, beta, w_alpha, w_beta):
    #     print(f"{t:>10.0f}{a:>16.3f}{b:>16.3f}{wa:>14.3f}{wb:>12.3f}")