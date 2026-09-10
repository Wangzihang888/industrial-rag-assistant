from app.config import (
    DENSE_TOP_K,
    BM25_TOP_K,
    RRF_K,
    RERANK_CANDIDATE_K,
)

from app.retrieval.rrf import (
    reciprocal_rank_fusion,
)


class HybridRetriever:

    def __init__(
        self,
        dense_retriever,
        bm25_retriever,
    ):
        """
        Hybrid Retriever

        组合：
        1. Dense Retriever
        2. BM25 Retriever
        3. RRF
        """

        self.dense_retriever = (
            dense_retriever
        )

        self.bm25_retriever = (
            bm25_retriever
        )


    def retrieve(
        self,
        query,
        dense_top_k=DENSE_TOP_K,
        bm25_top_k=BM25_TOP_K,
        candidate_k=RERANK_CANDIDATE_K,
    ):
        """
        Hybrid Retrieval 流程：

        Query
          ↓
        ┌───────────────┐
        │               │
      Dense           BM25
        │               │
        ↓               ↓
      Ranking         Ranking
        │               │
        └──────┬────────┘
               ↓
              RRF
               ↓
        Hybrid Ranking
        """

        # 1. Dense Retrieval
        dense_results = (
            self.dense_retriever
            .retrieve(
                query=query,
                top_k=dense_top_k,
            )
        )


        # 2. BM25 Retrieval
        bm25_results = (
            self.bm25_retriever
            .retrieve(
                query=query,
                top_k=bm25_top_k,
            )
        )


        # 3. 提取 Dense 排名
        dense_ranking = [
            result["chunk_id"]
            for result
            in dense_results
        ]


        # 4. 提取 BM25 排名
        bm25_ranking = [
            result["chunk_id"]
            for result
            in bm25_results
        ]


        # 5. RRF 融合
        rrf_results = (
            reciprocal_rank_fusion(
                rankings=[
                    dense_ranking,
                    bm25_ranking,
                ],
                k=RRF_K,
            )
        )


        # 6. 截取候选 Chunk
        hybrid_results = [
            {
                "chunk_id":
                    chunk_id,

                "rrf_score":
                    float(
                        rrf_score
                    ),
            }

            for chunk_id, rrf_score
            in rrf_results[
                :candidate_k
            ]
        ]


        return hybrid_results