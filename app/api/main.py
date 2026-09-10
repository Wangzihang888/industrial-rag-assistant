import logging

from contextlib import asynccontextmanager

from fastapi import (
    FastAPI,
    Request,
)

from fastapi.responses import (
    FileResponse,
    JSONResponse,
)

from app.bootstrap import (
    create_rag_pipeline,
)

from app.api.schemas import (
    ChatRequest,
    ChatResponse,
    ErrorResponse,
)

from app.exceptions import (
    RAGServiceError,
)


# ============================================================
# Logging 配置
# ============================================================

logging.basicConfig(
    level=logging.INFO,

    format=(
        "%(asctime)s | "
        "%(levelname)s | "
        "%(name)s | "
        "%(message)s"
    ),
)


logger = logging.getLogger(
    __name__
)


# ============================================================
# 全局对象
# ============================================================

rag_pipeline = None
qdrant_client = None


# ============================================================
# FastAPI 生命周期
# ============================================================

@asynccontextmanager
async def lifespan(
    app: FastAPI,
):

    global rag_pipeline
    global qdrant_client

    # --------------------------------------------------------
    # 应用启动
    # --------------------------------------------------------

    rag_pipeline, qdrant_client = (
        create_rag_pipeline()
    )

    logger.info(
        "RAG Pipeline initialized."
    )

    # yield 之前：
    # FastAPI 启动阶段
    #
    # yield：
    # 应用正式开始运行
    #
    # yield 之后：
    # FastAPI 关闭阶段

    yield

    # --------------------------------------------------------
    # 应用关闭
    # --------------------------------------------------------

    if qdrant_client is not None:

        qdrant_client.close()

    logger.info(
        "Qdrant client closed."
    )


# ============================================================
# 创建 FastAPI 应用
# ============================================================

app = FastAPI(
    title="Industrial RAG Assistant",

    description=(
        "基于 Hybrid Retrieval、Reranker、"
        "Evidence Gate 和 Qwen 的"
        "工业设备运维 RAG API"
    ),

    version="1.0.0",

    lifespan=lifespan,
)


# ============================================================
# RAG 自定义异常处理器
# ============================================================

@app.exception_handler(
    RAGServiceError
)
async def rag_service_error_handler(
    request: Request,
    exc: RAGServiceError,
):

    # logger.exception()
    # 会记录：
    #
    # 1. 错误信息
    # 2. traceback
    # 3. raise ... from exc 保留下来的异常链

    logger.exception(
        "RAG service failed: %s",
        exc,
    )

    return JSONResponse(
        status_code=503,

        content={
            "error":
                "RAG_SERVICE_ERROR",

            "message":
                "RAG 服务暂时不可用，"
                "请稍后重试。",
        },
    )


# ============================================================
# 未知异常处理器
# ============================================================

@app.exception_handler(
    Exception
)
async def unexpected_error_handler(
    request: Request,
    exc: Exception,
):

    logger.exception(
        "Unexpected server error: %s",
        exc,
    )

    return JSONResponse(
        status_code=500,

        content={
            "error":
                "INTERNAL_SERVER_ERROR",

            "message":
                "服务器内部发生异常。",
        },
    )


# ============================================================
# Web 首页
# ============================================================

@app.get(
    "/",
    tags=["Web"],
)
def home():

    return FileResponse(
        "app/static/index.html"
    )


# ============================================================
# 健康检查
# ============================================================

@app.get(
    "/health",
    tags=["System"],
)
def health_check():

    return {
        "status": "ok"
    }


# ============================================================
# RAG Chat API
# ============================================================

@app.post(
    "/api/chat",

    response_model=ChatResponse,

    responses={
        500: {
            "model":
                ErrorResponse,

            "description":
                "服务器内部错误",
        },

        503: {
            "model":
                ErrorResponse,

            "description":
                "RAG 服务不可用",
        },
    },

    tags=["RAG"],
)
def chat(
    request: ChatRequest,
):

    result = rag_pipeline.answer(
        request.query
    )

    return result