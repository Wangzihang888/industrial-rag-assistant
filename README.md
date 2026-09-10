# Industrial RAG Assistant

面向工业设备运维场景的知识库 RAG（Retrieval-Augmented Generation）问答系统。

项目基于 **Dense Retrieval + BM25 + RRF + Qwen Reranker + Evidence Gate + Qwen LLM** 构建完整 RAG Pipeline，支持工业设备故障知识检索、混合召回、候选重排、证据不足拒答、来源引用、离线评估、阶段延迟日志、FastAPI 服务化、Web UI 以及 Docker 容器化部署。

本项目主要用于学习和验证企业 RAG 应用的核心工程流程，不代表生产级系统。

---

# 1. Project Overview

工业设备运维知识通常分散在设备手册、故障说明、维护记录等资料中。

传统关键词搜索在面对语义改写时可能召回不足，而纯向量检索在处理设备故障码、报警编号等精确标识符时也存在局限。

例如工业知识中经常出现：

```text
E-103
AXIS-4
PLC-ALARM-003
```

因此，本项目采用：

```text
Dense Retrieval
+
BM25
+
Reciprocal Rank Fusion
+
Reranker
```

构建两阶段混合检索系统。

同时增加 Evidence Gate，在检索证据不足时直接拒答，降低大模型在知识库缺乏证据时生成无依据答案的风险。

---

# 2. Core Features

项目目前实现：

- Dense Semantic Retrieval
- BM25 Lexical Retrieval
- Industrial Identifier Tokenization
- Hybrid Retrieval
- Reciprocal Rank Fusion（RRF）
- Qwen Reranker
- Evidence Gate
- Grounded Answer Generation
- Source Citation
- Qdrant Vector Index
- Reproducible Knowledge Index Building
- Retrieval Evaluation
- Answerability Evaluation
- Stage-Level Latency Logging
- FastAPI REST API
- Swagger API Documentation
- Web UI
- Docker Containerization
- Docker Volume Persistent Qdrant Storage

---

# 3. System Architecture

```text
                         User
                           │
                           ▼
                       Web UI
                           │
                           ▼
                        FastAPI
                           │
                           ▼
                      RAGPipeline
                           │
                           ▼
                  Hybrid Retrieval
                  /              \
                 /                \
                ▼                  ▼
        Dense Retrieval        BM25 Retrieval
                │
                ▼
       Qwen Embedding
                │
                ▼
             Qdrant
                 \                /
                  \              /
                   ▼            ▼
                         RRF
                          │
                          ▼
                    Candidate Set
                          │
                          ▼
                    Qwen Reranker
                          │
                          ▼
                    Evidence Gate
                     /          \
                    /            \
                   ▼              ▼
               REJECT           ACCEPT
                                  │
                                  ▼
                           Context Builder
                                  │
                                  ▼
                              Qwen LLM
                                  │
                                  ▼
                         Answer + Sources
```

---

# 4. RAG Pipeline

完整查询流程：

```text
User Query
    ↓
Dense Retrieval
    +
BM25 Retrieval
    ↓
RRF Fusion
    ↓
Candidate Chunks
    ↓
Qwen Reranker
    ↓
Evidence Gate
    ↓
 ┌───────────────┐
 │               │
REJECT          ACCEPT
 │               │
固定拒答      Top-K Context
                 ↓
              Qwen LLM
                 ↓
          Answer + Sources
```

---

# 5. Technology Stack

| Category | Technology |
|---|---|
| Programming Language | Python 3.14 |
| API Framework | FastAPI |
| API Server | Uvicorn |
| Embedding Model | Qwen3.7 Text Embedding |
| Reranker | Qwen3.7 Text Rerank |
| LLM | Qwen |
| Vector Database | Qdrant |
| Lexical Retrieval | BM25 |
| Chinese Tokenization | jieba |
| Fusion Algorithm | Reciprocal Rank Fusion |
| API Client | OpenAI-compatible SDK / Requests |
| Configuration | python-dotenv |
| Frontend | HTML / CSS / JavaScript |
| Containerization | Docker |

---

# 6. Project Structure

```text
industrial-rag-assistant/
│
├── app/
│   ├── __init__.py
│   ├── config.py
│   ├── bootstrap.py
│   │
│   ├── api/
│   │   ├── __init__.py
│   │   ├── main.py
│   │   └── schemas.py
│   │
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── tokenizer.py
│   │   ├── dense.py
│   │   ├── bm25.py
│   │   ├── rrf.py
│   │   └── hybrid.py
│   │
│   ├── reranking/
│   │   ├── __init__.py
│   │   └── reranker.py
│   │
│   ├── generation/
│   │   ├── __init__.py
│   │   ├── context.py
│   │   └── generator.py
│   │
│   ├── pipeline/
│   │   ├── __init__.py
│   │   ├── evidence_gate.py
│   │   └── rag_pipeline.py
│   │
│   └── static/
│       └── index.html
│
├── data/
│   └── industrial_knowledge.jsonl
│
├── evaluation/
│   ├── __init__.py
│   ├── datasets.py
│   ├── retrieval_eval.py
│   ├── answerability_eval.py
│   └── results/
│       └── evaluation_results.md
│
├── scripts/
│   └── build_index.py
│
├── .dockerignore
├── .env.example
├── .gitignore
├── Dockerfile
├── requirements.txt
└── README.md
```

---

# 7. Retrieval Design

## 7.1 Dense Retrieval

Dense Retrieval 使用 Qwen Embedding 将：

```text
User Query
```

转换为：

```text
1024-dimensional embedding
```

然后通过 Qdrant 进行 Cosine Similarity Search。

Dense Retrieval 主要用于处理语义相似但词面不同的问题。

例如：

```text
用户：
控制柜太热应该排查什么？

知识库：
控制柜温度过高时，应检查……
```

虽然“太热”和“温度过高”词面不同，但 Dense Retrieval 可以利用语义表示进行召回。

---

# 8. BM25 Retrieval

BM25 用于补充 Dense Retrieval 对精确关键词和工业标识符的检索能力。

例如：

```text
E-103
AXIS-4
PLC-ALARM-003
```

项目实现工业文本 Tokenizer，尽量保留完整工业故障代码，再结合 jieba 对中文文本进行分词。

BM25 更擅长：

```text
Exact Match
Keyword Match
Fault Code Retrieval
```

---

# 9. Hybrid Retrieval

Dense Retrieval 与 BM25 各有优势：

```text
Dense
→ Semantic Recall

BM25
→ Lexical / Exact Match
```

因此项目采用 Hybrid Retrieval：

```text
Query
 ├── Dense Retrieval
 └── BM25 Retrieval
          ↓
         RRF
```

避免完全依赖单一检索方式。

---

# 10. Reciprocal Rank Fusion

Dense Similarity Score 和 BM25 Score 属于不同评分空间，因此项目不直接将两种 Score 相加。

使用 Reciprocal Rank Fusion：

```text
RRFScore(d) = Σ 1 / (k + rank(d))
```

当前：

```text
k = 60
```

RRF 基于不同检索器的排名进行融合，从而避免直接比较不同量纲的原始 Score。

---

# 11. Reranking

Hybrid Retrieval 负责提高候选召回能力，但候选集合仍可能包含相关性较弱的 Chunk。

因此项目增加第二阶段：

```text
Hybrid Retrieval
        ↓
Candidate Chunks
        ↓
Qwen Reranker
        ↓
Final Top-K
```

Reranker 同时考虑：

```text
Query + Candidate Document
```

重新计算候选文档与 Query 的相关性。

需要注意：

Rerank Score 表示相关性评分，不应直接解释为答案正确概率。

---

# 12. Evidence Gate

Top-K Retrieval 始终会返回候选结果，即使知识库中实际上不存在答案。

例如：

```text
润滑油品牌是什么？
```

知识库可能包含：

```text
润滑检查
润滑不足
```

但并没有提供具体润滑油品牌。

如果直接将 Top-K 交给 LLM，模型可能生成没有知识库证据支持的答案。

因此项目增加 Evidence Gate：

```text
Reranker
    ↓
Top1 Rerank Score
    ↓
Evidence Threshold
    ↓
 ┌───────────┐
 │           │
低于阈值    高于阈值
 │           │
REJECT      ACCEPT
```

当前开发阶段阈值：

```text
0.8
```

该阈值基于当前小规模评估集进行初步选择，不代表通用或生产级最优阈值。

---

# 13. Grounded Generation

只有 Evidence Gate 判断证据充分时，系统才调用 LLM。

Generation Prompt 要求：

1. 只能根据提供的 Context 回答；
2. 不允许使用 Context 之外的信息补充答案；
3. 缺少证据时不能猜测；
4. 关键结论需要引用对应 Source；
5. 不允许引用不存在的 Source。

Context 示例：

```text
[Source 1]
chunk_id: servo_e103
source: industrial_knowledge.jsonl
equipment: ABB IRB6700
category: servo
text: E-103 表示伺服驱动器过温故障……
```

最终输出：

```text
Answer
+
Sources
```

---

# 14. Knowledge Index

项目使用：

```text
data/industrial_knowledge.jsonl
```

作为当前知识库的 Source of Truth。

Qdrant Index 属于可重建的派生数据：

```text
industrial_knowledge.jsonl
        ↓
scripts/build_index.py
        ↓
Qwen Embedding
        ↓
Qdrant Points
```

因此：

```text
qdrant_data/
```

不会提交到 Git Repository。

---

# 15. Reproducible Index Building

执行：

```bash
python -m scripts.build_index
```

索引构建流程：

```text
Load Knowledge Records
        ↓
Validate Data
        ↓
Reset Qdrant Storage
        ↓
Create Collection
        ↓
Generate Embeddings
        ↓
Build Points
        ↓
Upsert
        ↓
Integrity Check
```

当前知识库：

```text
Knowledge Records: 20
Embedding Dimension: 1024
Indexed Points: 20
```

索引脚本会验证：

```text
Knowledge Records
=
Generated Embeddings
=
Built Points
=
Indexed Points
```

避免索引不完整但程序仍继续运行。

---

# 16. Stable Chunk Identity

项目使用：

```text
chunk_id
```

作为不同检索模块之间共享的业务层稳定标识。

例如：

```text
servo_e103
cabinet_cooling
plc_alarm_003
```

Qdrant Point ID 则根据 `chunk_id` 使用确定性 UUID 生成。

这样同一个 `chunk_id` 可以稳定映射到同一个数据库层 Point ID。

---

# 17. Evaluation

项目实现两类离线评估：

```text
Retrieval Evaluation
Answerability Evaluation
```

## Retrieval Evaluation

运行：

```bash
python -m evaluation.retrieval_eval
```

当前测试集包含 17 个 Retrieval Query。

结果：

| Method | Top1 Accuracy | Recall@3 |
|---|---:|---:|
| Dense | 100.00% | 100.00% |
| BM25 | 94.12% | 100.00% |
| Hybrid | 100.00% | 100.00% |
| Hybrid + Reranker | 100.00% | 100.00% |

BM25 当前存在一个典型 Top1 Failure Case：

```text
Query:
控制柜太热应该排查什么？

Expected:
cabinet_cooling

BM25 Top1:
cabinet_power
```

该案例体现了纯词法检索处理语义改写时的局限，也是引入 Dense Retrieval 和 Hybrid Retrieval 的设计依据之一。

---

# 18. Answerability Evaluation

运行：

```bash
python -m evaluation.answerability_eval
```

当前评估集：

```text
Total: 13
Correct: 13
Accuracy: 100%
False Accept: 0
False Reject: 0
```

测试同时包含：

```text
Answerable Queries
+
Unanswerable Queries
```

例如：

```text
Answerable:
E-103 是什么故障？

Unanswerable:
润滑油品牌是什么？
```

Evidence Gate 当前能够在该小规模测试集上区分两类 Query。

> Important: 当前结果基于仅 20 个知识 Chunk 和有限的人工构造评估 Query，仅用于开发阶段比较不同检索策略、验证 Pipeline 和进行 Regression Testing。100% 的结果不代表生产环境准确率，也不代表系统在真实开放场景中的泛化能力。

---

# 19. Logging and Latency Monitoring

系统记录主要 RAG 阶段耗时，包括：

```text
Hybrid Retrieval
Reranking
Generation
Total Request
```

示例：

```text
RAG request started
Hybrid retrieval completed | latency_ms=...
Reranking completed | latency_ms=...
Generation completed | latency_ms=...
RAG request completed | total_ms=...
```

阶段级 Latency Logging 可以帮助定位性能瓶颈。

例如：

```text
Retrieval Latency ↑
→ 检查 Embedding API / Qdrant

Reranking Latency ↑
→ 检查 Reranker API

Generation Latency ↑
→ 检查 LLM API
```

对于 Evidence Gate REJECT 的请求，系统不会继续调用 Generation LLM，因此可以减少无意义生成以及对应的延迟和模型调用成本。

---

# 20. FastAPI Service

启动：

```bash
python -m uvicorn app.api.main:app --reload
```

API：

```text
GET  /health
POST /api/chat
```

Swagger：

```text
http://localhost:8000/docs
```

FastAPI 使用 application lifespan 管理长期资源：

```text
Application Startup
        ↓
create_rag_pipeline()
        ↓
RAG Pipeline Ready
        ↓
Serve Requests
        ↓
Application Shutdown
        ↓
qdrant_client.close()
```

RAG Pipeline 在应用启动阶段初始化一次，并在多个请求之间复用，避免每次请求重复初始化 BM25、Qdrant Client 和其他组件。

---

# 21. Web UI

项目提供简单 Web UI：

```text
app/static/index.html
```

支持：

- Query 输入
- 示例问题
- Loading State
- ACCEPT / REJECT 状态
- Top1 Score
- Evidence Threshold
- Answer
- Source Cards

访问：

```text
http://localhost:8000
```

推荐测试：

```text
E-103 是什么故障？
```

用于测试 ACCEPT。

以及：

```text
润滑油品牌是什么？
```

用于测试 Evidence Gate REJECT。

---

# 22. Local Quick Start

## 22.1 Clone Repository

```bash
git clone <your-repository-url>
cd industrial-rag-assistant
```

## 22.2 Create Virtual Environment

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

## 22.3 Install Dependencies

```bash
python -m pip install -r requirements.txt
```

## 22.4 Configure Environment

根据：

```text
.env.example
```

创建：

```text
.env
```

需要配置：

```text
DASHSCOPE_API_KEY
DASHSCOPE_BASE_URL
DASHSCOPE_RERANK_URL
DASHSCOPE_CHAT_MODEL
QDRANT_PATH
```

不要将真实 `.env` 提交到 Git。

## 22.5 Build Knowledge Index

```bash
python -m scripts.build_index
```

## 22.6 Run Evaluation

```bash
python -m evaluation.retrieval_eval
python -m evaluation.answerability_eval
```

## 22.7 Start Application

```bash
python -m uvicorn app.api.main:app --reload
```

打开：

```text
Web UI:
http://localhost:8000

Swagger:
http://localhost:8000/docs

Health:
http://localhost:8000/health
```

---

# 23. Docker Deployment

项目支持 Docker 容器化运行。

当前架构采用：

```text
Docker Image
    ↓
FastAPI Container
    ↓
Qdrant Local Mode
    ↓
Docker Volume
```

Qdrant 数据通过 Docker Volume 持久化，使索引生命周期与应用 Container 生命周期解耦。

## 23.1 Build Docker Image

```bash
docker build -t industrial-rag-assistant:1.0 .
```

## 23.2 Create Persistent Qdrant Volume

```bash
docker volume create industrial-rag-qdrant
```

## 23.3 Build Knowledge Index in Docker

首先根据 `.env.example` 创建本地 `.env`。

Linux/macOS Shell:

```bash
docker run --rm \
  --env-file .env \
  -e QDRANT_PATH=/data/qdrant \
  -v industrial-rag-qdrant:/data/qdrant \
  industrial-rag-assistant:1.0 \
  python -m scripts.build_index
```

Windows PowerShell 可以直接使用单行：

```powershell
docker run --rm --env-file .env -e QDRANT_PATH=/data/qdrant -v industrial-rag-qdrant:/data/qdrant industrial-rag-assistant:1.0 python -m scripts.build_index
```

成功后应看到类似：

```text
Knowledge Records: 20
Generated Embeddings: 20
Indexed Points: 20
Embedding Dimension: 1024
```

## 23.4 Start FastAPI Container

Linux/macOS Shell:

```bash
docker run \
  --name industrial-rag-api \
  --env-file .env \
  -e QDRANT_PATH=/data/qdrant \
  -v industrial-rag-qdrant:/data/qdrant \
  -p 8000:8000 \
  industrial-rag-assistant:1.0
```

Windows PowerShell:

```powershell
docker run --name industrial-rag-api --env-file .env -e QDRANT_PATH=/data/qdrant -v industrial-rag-qdrant:/data/qdrant -p 8000:8000 industrial-rag-assistant:1.0
```

然后访问：

```text
Web UI:
http://localhost:8000

API Docs:
http://localhost:8000/docs

Health Check:
http://localhost:8000/health
```

---

# 24. Docker Container Management

查看正在运行的 Container：

```bash
docker ps
```

查看全部 Container：

```bash
docker ps -a
```

停止：

```bash
docker stop industrial-rag-api
```

重新启动：

```bash
docker start industrial-rag-api
```

查看日志：

```bash
docker logs industrial-rag-api
```

持续查看日志：

```bash
docker logs -f industrial-rag-api
```

查看 Image：

```bash
docker images
```

查看 Volume：

```bash
docker volume ls
```

项目不会将真实 API Key 写入 Docker Image。

Secret 在 Container Runtime 通过：

```text
--env-file .env
```

注入。

---

# 25. Docker Data Design

以下内容不会直接打入 Docker Image：

```text
.env
.venv/
qdrant_data/
.git/
logs/
__pycache__/
```

Docker Image 主要保存：

```text
Application Code
+
Python Runtime
+
Python Dependencies
+
Knowledge Source Data
+
Index Building Script
```

运行时数据与 Secret 分离：

```text
Application
→ Docker Image

Runtime Instance
→ Container

Qdrant Index
→ Docker Volume

API Key / Runtime Configuration
→ Environment Variables
```

---

# 26. Configuration Design

项目使用：

```text
.env
+
config.py
```

管理配置。

本地默认：

```text
QDRANT_PATH=qdrant_data
```

Docker 环境：

```text
QDRANT_PATH=/data/qdrant
```

同一份业务代码可以通过环境变量适配不同运行环境，而不需要针对本地和 Docker 修改核心代码。

---

# 27. Dependency Injection and Bootstrap

`bootstrap.py` 负责集中创建和连接系统组件：

```text
OpenAI-compatible Client
QdrantClient
BM25Retriever
DenseRetriever
HybridRetriever
QwenReranker
EvidenceGate
QwenGenerator
RAGPipeline
```

组件依赖在外部创建后通过构造函数注入。

例如：

```text
DenseRetriever
    ↑
Embedding Client
Qdrant Client
```

以及：

```text
RAGPipeline
    ↑
HybridRetriever
Reranker
EvidenceGate
Generator
```

这样可以降低组件之间的直接耦合，并使不同实现更容易替换和测试。

---

# 28. Security Considerations

项目不会将真实 Secret 提交到 Git Repository 或 Docker Image。

`.env` 被：

```text
.gitignore
```

排除，避免进入 Git。

同时被：

```text
.dockerignore
```

排除，避免进入 Docker Build Context / Image。

仓库只提供：

```text
.env.example
```

作为配置模板。

如果真实 API Key 曾经被提交到 Git 历史，应立即 Rotate / Revoke 对应 Key，并根据实际情况清理 Git 历史。

---

# 29. Limitations

当前项目仍存在以下限制：

1. 知识库规模仅为 20 个 Chunk；
2. Retrieval Evaluation 仅包含 17 个测试 Query；
3. Answerability Evaluation 仅包含 13 个测试 Query；
4. Evaluation Dataset 主要为人工构造；
5. Qdrant 当前使用 Local Mode；
6. 索引构建采用全量重建；
7. BM25 文档在应用启动阶段加载到内存；
8. 当前主要针对单机、小规模 Demo 场景；
9. 当前 Evidence Threshold 仅针对现有评估数据进行初步选择；
10. 当前评估结果不能代表生产环境泛化能力。

---

# 30. Future Improvements

如果进一步向生产级系统扩展，可以考虑：

```text
Qdrant Server
Incremental Indexing
Batch Indexing
Metadata Filtering
Query Rewrite
Caching
Access Control
Authentication
Rate Limiting
Metrics
Distributed Tracing
Larger Evaluation Dataset
Hard Negative Evaluation
Real User Query Evaluation
MRR / nDCG
Online Evaluation
Multi-instance Deployment
```

对于大规模知识库，还需要进一步解决：

```text
Index Versioning
Document Update
Document Deletion
Incremental Synchronization
BM25 Scalability
Vector Database Backup
Monitoring
Permission Isolation
```

---

# 31. Project Highlights

本项目重点不是简单调用 LLM，而是实现完整 RAG 工程链路：

```text
Knowledge Source
      ↓
Reproducible Indexing
      ↓
Dense + BM25
      ↓
RRF
      ↓
Reranker
      ↓
Evidence Gate
      ↓
Grounded Generation
      ↓
Source Citation
      ↓
FastAPI
      ↓
Web UI
      ↓
Evaluation
      ↓
Logging
      ↓
Docker Deployment
```

通过该项目完成了从：

```text
RAG Prototype
```

到：

```text
Runnable RAG Application
```

的完整工程实现。

---

# 32. Disclaimer

This project is designed for learning, portfolio demonstration, and small-scale RAG engineering validation.

The current evaluation results are based on a small knowledge base and limited manually constructed evaluation datasets.

They should not be interpreted as production-level accuracy, reliability, safety, or generalization performance.