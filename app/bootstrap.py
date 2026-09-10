from openai import OpenAI
from qdrant_client import QdrantClient

from app.config import (
    DASHSCOPE_API_KEY,
    DASHSCOPE_BASE_URL,
    QDRANT_PATH,
    QDRANT_COLLECTION,
)

from app.retrieval.bm25 import (
    BM25Retriever,
)

from app.retrieval.dense import (
    DenseRetriever,
)

from app.retrieval.hybrid import (
    HybridRetriever,
)

from app.reranking.reranker import (
    QwenReranker,
)

from app.pipeline.evidence_gate import (
    EvidenceGate,
)

from app.generation.generator import (
    QwenGenerator,
)

from app.pipeline.rag_pipeline import (
    RAGPipeline,
)


def create_rag_pipeline():

    # ========================================================
    # 1. 创建百炼 Client
    # ========================================================

    bailian_client = OpenAI(
        api_key=DASHSCOPE_API_KEY,
        base_url=DASHSCOPE_BASE_URL,
    )


    # ========================================================
    # 2. 创建 Qdrant Client
    # ========================================================

    qdrant_client = QdrantClient(
        path=QDRANT_PATH
    )


    # ========================================================
    # 3. 加载知识库
    # ========================================================

    points, _ = qdrant_client.scroll(
        collection_name=QDRANT_COLLECTION,
        limit=1000,
        with_payload=True,
        with_vectors=False,
    )


    if not points:
        qdrant_client.close()

        raise RuntimeError(
            "Qdrant 知识库为空，"
            "请先执行知识库索引脚本。"
        )


    # ========================================================
    # 4. 构造 BM25 所需数据
    # ========================================================

    documents = [
        point.payload["text"]
        for point in points
    ]


    chunk_ids = [
        point.payload["chunk_id"]
        for point in points
    ]


    # ========================================================
    # 5. 构造 chunk_map
    # ========================================================

    chunk_map = {
        point.payload["chunk_id"]: point
        for point in points
    }


    # ========================================================
    # 6. 创建 BM25 Retriever
    # ========================================================

    bm25_retriever = BM25Retriever(
        documents=documents,
        chunk_ids=chunk_ids,
    )


    # ========================================================
    # 7. 创建 Dense Retriever
    # ========================================================

    dense_retriever = DenseRetriever(
        embedding_client=bailian_client,
        qdrant_client=qdrant_client,
    )


    # ========================================================
    # 8. 创建 Hybrid Retriever
    # ========================================================

    hybrid_retriever = HybridRetriever(
        dense_retriever=dense_retriever,
        bm25_retriever=bm25_retriever,
    )


    # ========================================================
    # 9. 创建 Reranker
    # ========================================================

    reranker = QwenReranker(
        chunk_map=chunk_map
    )


    # ========================================================
    # 10. 创建 Evidence Gate
    # ========================================================

    evidence_gate = EvidenceGate()


    # ========================================================
    # 11. 创建 Generator
    # ========================================================

    generator = QwenGenerator(
        llm_client=bailian_client
    )


    # ========================================================
    # 12. 创建完整 Pipeline
    # ========================================================

    rag_pipeline = RAGPipeline(
        hybrid_retriever=hybrid_retriever,
        reranker=reranker,
        evidence_gate=evidence_gate,
        generator=generator,
    )


    return (
        rag_pipeline,
        qdrant_client,
    )