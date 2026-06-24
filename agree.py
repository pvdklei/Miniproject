"""

Overlap measures for comparing two rankings of items, used to check whether two
attribution methods agree on which sparse dimensions or image pixels matter 
(research question 1).

Each method gives a score per dimension. We compare two methods by how similar
their rankings of the dimensions are. The measures here are:

kendall_tau      : rank correlation over all dimensions, in [-1, 1].
overlap_at_k     : fraction of the top-k that the two rankings share, in [0, 1].
jaccard_at_k     : intersection over union of the two top-k sets, in [0, 1].
rank_biased_overlap : overlap of the top of the rankings, weighted so earlier
                      ranks count more, in [0, 1].

"""

import numpy as np


def _top_k_set(scores: np.ndarray, k: int) -> set[int]:
    """Indices of the k highest scores."""
    return set(np.argsort(-scores)[:k].tolist())


def kendall_tau(scores_a: np.ndarray, scores_b: np.ndarray, k: int = 100) -> float:
    """Kendall's tau rank correlation between two score vectors.

    Counts pairs of dimensions ordered the same way by both methods (concordant)
    minus the pairs ordered oppositely (discordant), over all pairs. +1 means the
    same ranking, -1 the reverse, 0 no relation.

    There are 40k dimensions but almost all have a tiny score, so comparing every
    pair is both slow and dominated by noise. We restrict to the union of the two
    methods' top-k dimensions, which is where the ranking actually carries meaning.
    """
    dims = sorted(_top_k_set(scores_a, k) | _top_k_set(scores_b, k))
    a, b = scores_a[dims], scores_b[dims]
    m = len(a)
    concordant = discordant = 0
    for i in range(m):
        for j in range(i + 1, m):
            sign = np.sign(a[i] - a[j]) * np.sign(b[i] - b[j])
            if sign > 0:
                concordant += 1
            elif sign < 0:
                discordant += 1
    total = concordant + discordant
    return (concordant - discordant) / total if total else 0.0


def overlap_at_k(scores_a: np.ndarray, scores_b: np.ndarray, k: int) -> float:
    """Fraction of the top-k dimensions that both methods share."""
    shared = _top_k_set(scores_a, k) & _top_k_set(scores_b, k)
    return len(shared) / k


def jaccard_at_k(scores_a: np.ndarray, scores_b: np.ndarray, k: int) -> float:
    """Intersection over union of the two top-k sets.

    (idea3.md wrote union/intersection, but jaccard is intersection/union; that is
    what we use so the value stays in [0, 1].)
    """
    top_a, top_b = _top_k_set(scores_a, k), _top_k_set(scores_b, k)
    union = top_a | top_b
    return len(top_a & top_b) / len(union) if union else 0.0


def rank_biased_overlap(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    p: float = 0.9,
    depth: int = 100,
) -> float:
    """Rank biased overlap: agreement of the top of the rankings.

    Walks down both rankings and at each depth k measures overlap_at_k, then
    averages those with weights p^(k-1) so the very top matters most. p sets how
    quickly the weight falls off (higher p looks deeper).
    """
    rank_a = np.argsort(-scores_a)
    rank_b = np.argsort(-scores_b)
    depth = min(depth, len(scores_a))
    seen_a: set[int] = set()
    seen_b: set[int] = set()
    total = 0.0
    for k in range(1, depth + 1):
        seen_a.add(int(rank_a[k - 1]))
        seen_b.add(int(rank_b[k - 1]))
        agreement = len(seen_a & seen_b) / k  # overlap@k
        total += (p ** (k - 1)) * agreement
    return (1 - p) * total


def all_measures(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    k: int = 25,
    rbo_p: float = 0.9,
    rbo_depth: int = 100,
) -> dict[str, float]:
    """Run every overlap measure on a pair of score vectors and return the scores.

    k is the cutoff for kendall_tau, overlap_at_k and jaccard_at_k. rank_biased_overlap
    instead walks down to rbo_depth with weight rbo_p, so it has no hard cutoff.
    """
    return {
        "kendall_tau": kendall_tau(scores_a, scores_b, k),
        "overlap_at_k": overlap_at_k(scores_a, scores_b, k),
        "jaccard_at_k": jaccard_at_k(scores_a, scores_b, k),
        "rank_biased_overlap": rank_biased_overlap(
            scores_a, scores_b, rbo_p, rbo_depth
        ),
    }


MEASURE_NAMES = ["kendall_tau", "overlap_at_k", "jaccard_at_k", "rank_biased_overlap"]
