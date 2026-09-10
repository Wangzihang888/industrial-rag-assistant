from app.config import (
    EMBEDDING_MODEL,
    EMBEDDING_DIMENSION,
    QDRANT_COLLECTION,
)

from app.exceptions import (
    RAGServiceError,
)


class DenseRetriever:

    def __init__(
        self,
        embedding_client,
        qdrant_client,
    ):
        self.embedding_client = (
            embedding_client
        )

        self.qdrant_client = (
            qdrant_client
        )


    def retrieve(
        self,
        query,
        top_k=10,
    ):

        # ====================================================
        # 1. 调用 Embedding API
        # ====================================================

        try:

            response = (
                self.embedding_client
                .embeddings
                .create(
                    model=EMBEDDING_MODEL,
                    input=query,
                    dimensions=
                        EMBEDDING_DIMENSION,
                    encoding_format=
                        "float",
                )
            )

            query_embedding = (
                response
                .data[0]
                .embedding
            )

        except Exception as exc:

            raise RAGServiceError(
                "Embedding API 调用失败"
            ) from exc


        # ====================================================
        # 2. 查询 Qdrant
        # ====================================================

        try:

            points = (
                self.qdrant_client
                .query_points(
                    collection_name=
                        QDRANT_COLLECTION,

                    query=
                        query_embedding,

                    limit=
                        top_k,

                    with_payload=
                        True,
                )
                .points
            )

        except Exception as exc:

            raise RAGServiceError(
                "Qdrant 向量检索失败"
            ) from exc


        # ====================================================
        # 3. 转换结果
        # ====================================================

        return [
            {
                "chunk_id":
                    point.payload[
                        "chunk_id"
                    ],

                "score":
                    float(
                        point.score
                    ),

                "text":
                    point.payload[
                        "text"
                    ],
            }

            for point in points
        ]