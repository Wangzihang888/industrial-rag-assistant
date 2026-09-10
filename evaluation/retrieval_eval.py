from app.bootstrap import (
    create_rag_pipeline,
)

from evaluation.datasets import (
    RETRIEVAL_TEST_CASES,
)


# ============================================================
# 计算 Top1 是否正确
# ============================================================

def calculate_top1(
    ranked_chunk_ids,
    expected_chunk_id,
):

    if not ranked_chunk_ids:
        return 0

    return int(
        ranked_chunk_ids[0]
        == expected_chunk_id
    )


# ============================================================
# 计算 Recall@K
# ============================================================

def calculate_recall_at_k(
    ranked_chunk_ids,
    expected_chunk_id,
    k=3,
):

    top_k = (
        ranked_chunk_ids[:k]
    )

    return int(
        expected_chunk_id
        in top_k
    )


# ============================================================
# 打印失败案例
# ============================================================

def print_failure_case(
    method_name,
    query,
    query_type,
    expected_chunk_id,
    ranked_chunk_ids,
    top_k=3,
):

    print()
    print(
        "=" * 70
    )

    print(
        f"[{method_name} FAILURE]"
    )

    print(
        "=" * 70
    )

    print(
        "Type:"
    )

    print(
        query_type
    )

    print()

    print(
        "Query:"
    )

    print(
        query
    )

    print()

    print(
        "Expected:"
    )

    print(
        expected_chunk_id
    )

    print()

    print(
        "Actual Top1:"
    )

    if ranked_chunk_ids:

        print(
            ranked_chunk_ids[0]
        )

    else:

        print(
            "NO RESULT"
        )

    print()

    print(
        f"Top{top_k}:"
    )

    for rank, chunk_id in enumerate(
        ranked_chunk_ids[:top_k],
        start=1,
    ):

        print(
            f"{rank}. {chunk_id}"
        )

    print()


# ============================================================
# 主函数
# ============================================================

def main():

    rag_pipeline, qdrant_client = (
        create_rag_pipeline()
    )

    try:

        # ====================================================
        # 指标累计器
        # ====================================================

        dense_top1_total = 0
        dense_recall3_total = 0

        bm25_top1_total = 0
        bm25_recall3_total = 0

        hybrid_top1_total = 0
        hybrid_recall3_total = 0

        rerank_top1_total = 0
        rerank_recall3_total = 0


        # ====================================================
        # Failure 计数
        # ====================================================

        dense_failures = 0
        bm25_failures = 0
        hybrid_failures = 0
        rerank_failures = 0


        # ====================================================
        # 遍历测试集
        # ====================================================

        for case in RETRIEVAL_TEST_CASES:

            query = (
                case["query"]
            )

            expected_chunk_id = (
                case[
                    "expected_chunk_id"
                ]
            )

            query_type = (
                case.get(
                    "type",
                    "unknown",
                )
            )


            # =================================================
            # 1. Dense Retrieval
            # =================================================

            dense_results = (
                rag_pipeline
                .hybrid_retriever
                .dense_retriever
                .retrieve(
                    query=query,
                    top_k=10,
                )
            )

            dense_ranking = [
                item["chunk_id"]

                for item
                in dense_results
            ]


            # =================================================
            # 2. BM25 Retrieval
            # =================================================

            bm25_results = (
                rag_pipeline
                .hybrid_retriever
                .bm25_retriever
                .retrieve(
                    query=query,
                    top_k=10,
                )
            )

            bm25_ranking = [
                item["chunk_id"]

                for item
                in bm25_results
            ]


            # =================================================
            # 3. Hybrid Retrieval
            # =================================================

            hybrid_results = (
                rag_pipeline
                .hybrid_retriever
                .retrieve(
                    query=query,

                    dense_top_k=10,
                    bm25_top_k=10,
                    candidate_k=10,
                )
            )

            hybrid_ranking = [
                item["chunk_id"]

                for item
                in hybrid_results
            ]


            # =================================================
            # 4. Hybrid + Reranker
            # =================================================

            reranked_results = (
                rag_pipeline
                .reranker
                .rerank(
                    query=query,

                    candidate_chunk_ids=
                        hybrid_ranking,

                    final_k=3,
                )
            )

            rerank_ranking = [
                item["chunk_id"]

                for item
                in reranked_results
            ]


            # =================================================
            # 5. Dense Metrics
            # =================================================

            dense_top1 = (
                calculate_top1(
                    dense_ranking,
                    expected_chunk_id,
                )
            )

            dense_recall3 = (
                calculate_recall_at_k(
                    dense_ranking,
                    expected_chunk_id,
                    k=3,
                )
            )


            dense_top1_total += (
                dense_top1
            )

            dense_recall3_total += (
                dense_recall3
            )


            # 如果 Dense Top1 错误
            if dense_top1 == 0:

                dense_failures += 1

                print_failure_case(
                    method_name=
                        "Dense",

                    query=
                        query,

                    query_type=
                        query_type,

                    expected_chunk_id=
                        expected_chunk_id,

                    ranked_chunk_ids=
                        dense_ranking,

                    top_k=3,
                )


            # =================================================
            # 6. BM25 Metrics
            # =================================================

            bm25_top1 = (
                calculate_top1(
                    bm25_ranking,
                    expected_chunk_id,
                )
            )

            bm25_recall3 = (
                calculate_recall_at_k(
                    bm25_ranking,
                    expected_chunk_id,
                    k=3,
                )
            )


            bm25_top1_total += (
                bm25_top1
            )

            bm25_recall3_total += (
                bm25_recall3
            )


            # 如果 BM25 Top1 错误
            if bm25_top1 == 0:

                bm25_failures += 1

                print_failure_case(
                    method_name=
                        "BM25",

                    query=
                        query,

                    query_type=
                        query_type,

                    expected_chunk_id=
                        expected_chunk_id,

                    ranked_chunk_ids=
                        bm25_ranking,

                    top_k=3,
                )


            # =================================================
            # 7. Hybrid Metrics
            # =================================================

            hybrid_top1 = (
                calculate_top1(
                    hybrid_ranking,
                    expected_chunk_id,
                )
            )

            hybrid_recall3 = (
                calculate_recall_at_k(
                    hybrid_ranking,
                    expected_chunk_id,
                    k=3,
                )
            )


            hybrid_top1_total += (
                hybrid_top1
            )

            hybrid_recall3_total += (
                hybrid_recall3
            )


            # 如果 Hybrid Top1 错误
            if hybrid_top1 == 0:

                hybrid_failures += 1

                print_failure_case(
                    method_name=
                        "Hybrid",

                    query=
                        query,

                    query_type=
                        query_type,

                    expected_chunk_id=
                        expected_chunk_id,

                    ranked_chunk_ids=
                        hybrid_ranking,

                    top_k=3,
                )


            # =================================================
            # 8. Reranker Metrics
            # =================================================

            rerank_top1 = (
                calculate_top1(
                    rerank_ranking,
                    expected_chunk_id,
                )
            )

            rerank_recall3 = (
                calculate_recall_at_k(
                    rerank_ranking,
                    expected_chunk_id,
                    k=3,
                )
            )


            rerank_top1_total += (
                rerank_top1
            )

            rerank_recall3_total += (
                rerank_recall3
            )


            # 如果 Reranker Top1 错误
            if rerank_top1 == 0:

                rerank_failures += 1

                print_failure_case(
                    method_name=
                        "Hybrid + Reranker",

                    query=
                        query,

                    query_type=
                        query_type,

                    expected_chunk_id=
                        expected_chunk_id,

                    ranked_chunk_ids=
                        rerank_ranking,

                    top_k=3,
                )


        # ====================================================
        # 总测试数量
        # ====================================================

        total = len(
            RETRIEVAL_TEST_CASES
        )


        # ====================================================
        # 最终结果
        # ====================================================

        print()
        print(
            "=" * 70
        )

        print(
            "Retrieval Evaluation Summary"
        )

        print(
            "=" * 70
        )

        print()

        print(
            "Test Cases:",
            total,
        )

        print()


        print(
            "Dense Top1:",
            f"{dense_top1_total / total:.2%}",
        )

        print(
            "Dense Recall@3:",
            f"{dense_recall3_total / total:.2%}",
        )

        print(
            "Dense Failures:",
            dense_failures,
        )


        print()


        print(
            "BM25 Top1:",
            f"{bm25_top1_total / total:.2%}",
        )

        print(
            "BM25 Recall@3:",
            f"{bm25_recall3_total / total:.2%}",
        )

        print(
            "BM25 Failures:",
            bm25_failures,
        )


        print()


        print(
            "Hybrid Top1:",
            f"{hybrid_top1_total / total:.2%}",
        )

        print(
            "Hybrid Recall@3:",
            f"{hybrid_recall3_total / total:.2%}",
        )

        print(
            "Hybrid Failures:",
            hybrid_failures,
        )


        print()


        print(
            "Hybrid + Reranker Top1:",
            f"{rerank_top1_total / total:.2%}",
        )

        print(
            "Hybrid + Reranker Recall@3:",
            f"{rerank_recall3_total / total:.2%}",
        )

        print(
            "Hybrid + Reranker Failures:",
            rerank_failures,
        )


    finally:

        qdrant_client.close()


if __name__ == "__main__":
    main()