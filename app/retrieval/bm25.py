from rank_bm25 import BM25Okapi

from app.retrieval.tokenizer import (
    industrial_tokenize,
)


class BM25Retriever:

    def __init__(
        self,
        documents,
        chunk_ids,
    ):
        """
        初始化 BM25 检索器。

        documents:
            所有知识块文本。

        chunk_ids:
            每个知识块对应的业务 chunk_id。
        """

        if len(documents) != len(chunk_ids):
            raise ValueError(
                "documents 和 chunk_ids "
                "数量必须一致"
            )

        self.documents = documents
        self.chunk_ids = chunk_ids

        self.tokenized_documents = [
            industrial_tokenize(document)
            for document in documents
        ]

        self.bm25 = BM25Okapi(
            self.tokenized_documents
        )


    def retrieve(
        self,
        query,
        top_k=10,
    ):
        """
        使用 BM25 检索与 query
        最相关的 Top-K Chunk。
        """

        tokenized_query = (
            industrial_tokenize(
                query
            )
        )

        scores = self.bm25.get_scores(
            tokenized_query
        )

        ranked_results = sorted(
            enumerate(scores),
            key=lambda x: x[1],
            reverse=True,
        )

        results = [
            {
                "chunk_id":
                    self.chunk_ids[
                        document_index
                    ],

                "score":
                    float(score),
            }
            for document_index, score
            in ranked_results[:top_k]
        ]

        return results