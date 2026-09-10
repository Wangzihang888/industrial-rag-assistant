import logging
import time

from app.config import (
    DENSE_TOP_K,
    BM25_TOP_K,
    RERANK_CANDIDATE_K,
    FINAL_TOP_K,
)


logger = logging.getLogger(
    __name__
)


class RAGPipeline:

    def __init__(
        self,
        hybrid_retriever,
        reranker,
        evidence_gate,
        generator,
    ):

        self.hybrid_retriever = (
            hybrid_retriever
        )

        self.reranker = reranker

        self.evidence_gate = (
            evidence_gate
        )

        self.generator = generator


    def answer(
        self,
        query,
        dense_top_k=DENSE_TOP_K,
        bm25_top_k=BM25_TOP_K,
        candidate_k=RERANK_CANDIDATE_K,
        final_k=FINAL_TOP_K,
    ):

        # ====================================================
        # 整个 RAG 请求开始计时
        # ====================================================

        total_start = (
            time.perf_counter()
        )

        logger.info(
            "RAG request started | query=%s",
            query,
        )


        # ====================================================
        # 1. Hybrid Retrieval
        # ====================================================

        retrieval_start = (
            time.perf_counter()
        )


        hybrid_results = (
            self.hybrid_retriever
            .retrieve(
                query=query,

                dense_top_k=
                    dense_top_k,

                bm25_top_k=
                    bm25_top_k,

                candidate_k=
                    candidate_k,
            )
        )


        retrieval_ms = (
            time.perf_counter()
            - retrieval_start
        ) * 1000


        logger.info(
            "Hybrid retrieval completed | "
            "latency_ms=%.2f | "
            "candidates=%s",

            retrieval_ms,
            len(hybrid_results),
        )


        # ====================================================
        # 2. Reranker
        # ====================================================

        candidate_chunk_ids = [
            result["chunk_id"]
            for result
            in hybrid_results
        ]


        rerank_start = (
            time.perf_counter()
        )


        reranked_results = (
            self.reranker
            .rerank(
                query=query,

                candidate_chunk_ids=
                    candidate_chunk_ids,

                final_k=
                    final_k,
            )
        )


        rerank_ms = (
            time.perf_counter()
            - rerank_start
        ) * 1000


        logger.info(
            "Reranking completed | "
            "latency_ms=%.2f | "
            "results=%s",

            rerank_ms,
            len(reranked_results),
        )


        # ====================================================
        # 3. Evidence Gate
        # ====================================================

        gate_result = (
            self.evidence_gate
            .check(
                reranked_results
            )
        )


        # ====================================================
        # 4. REJECT
        # ====================================================

        if not gate_result[
            "accepted"
        ]:

            total_ms = (
                time.perf_counter()
                - total_start
            ) * 1000


            logger.info(
                "RAG request rejected | "
                "top1_score=%.4f | "
                "threshold=%.4f | "
                "total_ms=%.2f",

                gate_result[
                    "top1_score"
                ],

                gate_result[
                    "threshold"
                ],

                total_ms,
            )


            return {
                "status":
                    "REJECT",

                "answer":
                    (
                        "根据当前知识库，"
                        "无法找到足够证据"
                        "回答该问题。"
                    ),

                "top1_score":
                    gate_result[
                        "top1_score"
                    ],

                "threshold":
                    gate_result[
                        "threshold"
                    ],

                "sources":
                    [],
            }


        # ====================================================
        # 5. LLM Generation
        # ====================================================

        generation_start = (
            time.perf_counter()
        )


        answer = (
            self.generator
            .generate(
                query=query,

                reranked_results=
                    reranked_results,
            )
        )


        generation_ms = (
            time.perf_counter()
            - generation_start
        ) * 1000


        logger.info(
            "Generation completed | "
            "latency_ms=%.2f",

            generation_ms,
        )


        # ====================================================
        # 6. Sources
        # ====================================================

        sources = [
            {
                "source_number":
                    index,

                "chunk_id":
                    item[
                        "chunk_id"
                    ],

                "source":
                    item.get(
                        "source",
                        "unknown",
                    ),

                "equipment":
                    item.get(
                        "equipment",
                        "unknown",
                    ),

                "category":
                    item.get(
                        "category",
                        "unknown",
                    ),

                "rerank_score":
                    item[
                        "rerank_score"
                    ],

                "text":
                    item[
                        "text"
                    ],
            }

            for index, item
            in enumerate(
                reranked_results,
                start=1,
            )
        ]


        # ====================================================
        # 7. 整体耗时
        # ====================================================

        total_ms = (
            time.perf_counter()
            - total_start
        ) * 1000


        logger.info(
            "RAG request completed | "
            "status=ACCEPT | "
            "total_ms=%.2f",

            total_ms,
        )


        # ====================================================
        # 8. 返回结果
        # ====================================================

        return {
            "status":
                "ACCEPT",

            "answer":
                answer,

            "top1_score":
                gate_result[
                    "top1_score"
                ],

            "threshold":
                gate_result[
                    "threshold"
                ],

            "sources":
                sources,
        }