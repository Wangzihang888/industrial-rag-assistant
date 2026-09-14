import os

from dotenv import load_dotenv


# ============================================================
# 1. 加载 .env
# ============================================================

load_dotenv()


# ============================================================
# 2. API 配置
# ============================================================

DASHSCOPE_API_KEY = os.getenv(
    "DASHSCOPE_API_KEY"
)

DASHSCOPE_BASE_URL = os.getenv(
    "DASHSCOPE_BASE_URL"
)

DASHSCOPE_RERANK_URL = os.getenv(
    "DASHSCOPE_RERANK_URL"
)


# ============================================================
# 3. 模型配置
# ============================================================

EMBEDDING_MODEL = (
    "qwen3.7-text-embedding"
)

RERANK_MODEL = (
    "qwen3.7-text-rerank"
)

LLM_MODEL = os.getenv(
    "DASHSCOPE_CHAT_MODEL",
    "qwen3.7-plus",
)


# ============================================================
# 4. Embedding 配置
# ============================================================

EMBEDDING_DIMENSION = 1024


# ============================================================
# 5. Qdrant 配置
# ============================================================

QDRANT_PATH = os.getenv(
    "QDRANT_PATH",
    "qdrant_data"
)

QDRANT_COLLECTION = (
    "industrial_manuals_v2"
)


# ============================================================
# 6. Retrieval 配置
# ============================================================

DENSE_TOP_K = 10

BM25_TOP_K = 10

RRF_K = 60

RERANK_CANDIDATE_K = 10

FINAL_TOP_K = 3


# ============================================================
# 7. Evidence Gate 配置
# ============================================================

EVIDENCE_THRESHOLD = 0.8


# ============================================================
# 8. 检查必要配置
# ============================================================

if not DASHSCOPE_API_KEY:
    raise ValueError(
        "没有找到 DASHSCOPE_API_KEY"
    )

if not DASHSCOPE_BASE_URL:
    raise ValueError(
        "没有找到 DASHSCOPE_BASE_URL"
    )

if not DASHSCOPE_RERANK_URL:
    raise ValueError(
        "没有找到 DASHSCOPE_RERANK_URL"
    )