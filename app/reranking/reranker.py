import requests

from app.config import (
    DASHSCOPE_API_KEY,
    DASHSCOPE_RERANK_URL,
    RERANK_MODEL,
    FINAL_TOP_K,
)

from app.exceptions import (
    RAGServiceError,
)


class QwenReranker:

    def __init__(
        self,
        chunk_map,
    ):
        self.chunk_map = chunk_map


    def rerank(
        self,
        query,
        candidate_chunk_ids,
        final_k=FINAL_TOP_K,
    ):

        # ====================================================
        # 1. 构造候选文档
        # ====================================================

        candidates = [
            {
                "chunk_id":
                    chunk_id,

                "text":
                    self.chunk_map[
                        chunk_id
                    ].payload[
                        "text"
                    ],

                "source":
                    self.chunk_map[
                        chunk_id
                    ].payload.get(
                        "source",
                        "unknown",
                    ),

                "equipment":
                    self.chunk_map[
                        chunk_id
                    ].payload.get(
                        "equipment",
                        "unknown",
                    ),

                "category":
                    self.chunk_map[
                        chunk_id
                    ].payload.get(
                        "category",
                        "unknown",
                    ),
            }

            for chunk_id
            in candidate_chunk_ids
        ]


        documents = [
            candidate["text"]
            for candidate
            in candidates
        ]


        # ====================================================
        # 2. 构造 HTTP 请求
        # ====================================================

        headers = {
            "Authorization":
                f"Bearer {DASHSCOPE_API_KEY}",

            "Content-Type":
                "application/json",
        }


        payload = {
            "model":
                RERANK_MODEL,

            "input": {
                "query":
                    query,

                "documents":
                    documents,
            },

            "parameters": {
                "top_n":
                    len(documents),

                "instruct":
                    (
                        "Given a search query, "
                        "rank the passages by "
                        "their relevance to "
                        "the query."
                    ),
            },
        }


        # ====================================================
        # 3. 调用 Reranker API
        # ====================================================

        try:

            response = requests.post(
                DASHSCOPE_RERANK_URL,
                headers=headers,
                json=payload,
                timeout=60,
            )

            response.raise_for_status()

            response_data = (
                response.json()
            )

            raw_results = (
                response_data[
                    "output"
                ][
                    "results"
                ]
            )

        except requests.RequestException as exc:

            raise RAGServiceError(
                "Reranker API 请求失败"
            ) from exc

        except (
            KeyError,
            TypeError,
            ValueError,
        ) as exc:

            raise RAGServiceError(
                "Reranker API 返回数据格式异常"
            ) from exc


        # ====================================================
        # 4. 映射 Rerank 结果
        # ====================================================

        reranked_results = []


        try:

            for result in raw_results:

                original_index = (
                    result["index"]
                )

                rerank_score = (
                    result[
                        "relevance_score"
                    ]
                )

                candidate = (
                    candidates[
                        original_index
                    ]
                )


                reranked_results.append(
                    {
                        "chunk_id":
                            candidate[
                                "chunk_id"
                            ],

                        "text":
                            candidate[
                                "text"
                            ],

                        "source":
                            candidate[
                                "source"
                            ],

                        "equipment":
                            candidate[
                                "equipment"
                            ],

                        "category":
                            candidate[
                                "category"
                            ],

                        "rerank_score":
                            float(
                                rerank_score
                            ),
                    }
                )

        except (
            KeyError,
            IndexError,
            TypeError,
            ValueError,
        ) as exc:

            raise RAGServiceError(
                "Reranker 结果解析失败"
            ) from exc


        return reranked_results[
            :final_k
        ]