def reciprocal_rank_fusion(
    rankings,
    k=60,
):
    """
    使用 Reciprocal Rank Fusion 融合多个排序结果。

    参数：
    rankings:
        一个二维列表，例如：
        [
            ["doc_a", "doc_b", "doc_c"],
            ["doc_b", "doc_a", "doc_d"],
        ]

    k:
        RRF 的平滑参数，默认 60。

    返回：
        [
            ("doc_a", rrf_score),
            ("doc_b", rrf_score),
            ...
        ]

        并按照 rrf_score 从高到低排序。
    """

    rrf_scores = {}

    for ranking in rankings:

        for rank, chunk_id in enumerate(
            ranking,
            start=1,
        ):

            score = 1 / (
                k + rank
            )

            rrf_scores[
                chunk_id
            ] = (
                rrf_scores.get(
                    chunk_id,
                    0,
                )
                + score
            )

    ranked_results = sorted(
        rrf_scores.items(),
        key=lambda x: x[1],
        reverse=True,
    )

    return ranked_results