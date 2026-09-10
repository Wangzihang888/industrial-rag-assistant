import json
import shutil
import uuid

from pathlib import Path

from openai import OpenAI

from qdrant_client import QdrantClient

from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
)

from app.config import (
    DASHSCOPE_API_KEY,
    DASHSCOPE_BASE_URL,
    EMBEDDING_MODEL,
    EMBEDDING_DIMENSION,
    QDRANT_PATH,
    QDRANT_COLLECTION,
)


# ============================================================
# 项目路径
# ============================================================

# scripts/build_index.py
#        ↓
# scripts/
#        ↓
# 项目根目录
PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)


# ============================================================
# 原始知识文件
# ============================================================

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "industrial_knowledge.jsonl"
)


# ============================================================
# Qdrant 本地数据库目录
# ============================================================

QDRANT_STORAGE_PATH = (
    PROJECT_ROOT
    / QDRANT_PATH
)


# ============================================================
# Source Metadata
# ============================================================

SOURCE_NAME = (
    DATA_PATH.name
)


# ============================================================
# 创建稳定的 Qdrant Point ID
# ============================================================

def create_point_id(
    chunk_id,
):

    # uuid5 是确定性 UUID。
    #
    # 相同 chunk_id：
    #
    # servo_e103
    #
    # 无论执行多少次，
    # 都会生成相同的 UUID。

    return str(
        uuid.uuid5(
            uuid.NAMESPACE_DNS,
            chunk_id,
        )
    )


# ============================================================
# 加载 JSONL
# ============================================================

def load_knowledge_records():

    records = []


    # --------------------------------------------------------
    # 先检查文件是否存在
    # --------------------------------------------------------

    if not DATA_PATH.exists():

        raise FileNotFoundError(
            (
                "找不到知识文件："
                f"{DATA_PATH}"
            )
        )


    # --------------------------------------------------------
    # 读取 JSONL
    # --------------------------------------------------------

    with DATA_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:

        for line_number, line in enumerate(
            file,
            start=1,
        ):

            line = line.strip()


            # 空行跳过
            if not line:
                continue


            # ------------------------------------------------
            # JSON String → Python dict
            # ------------------------------------------------

            try:

                record = json.loads(
                    line
                )

            except json.JSONDecodeError as exc:

                raise ValueError(
                    (
                        "JSONL 数据格式错误，"
                        f"line={line_number}"
                    )
                ) from exc


            # ------------------------------------------------
            # 检查必须字段
            # ------------------------------------------------

            required_fields = [
                "chunk_id",
                "text",
                "equipment",
                "category",
            ]


            for field in required_fields:

                if field not in record:

                    raise ValueError(
                        (
                            "知识数据缺少字段："
                            f"{field}，"
                            f"line={line_number}"
                        )
                    )


            # ------------------------------------------------
            # chunk_id 不能为空
            # ------------------------------------------------

            if not str(
                record["chunk_id"]
            ).strip():

                raise ValueError(
                    (
                        "chunk_id 不能为空，"
                        f"line={line_number}"
                    )
                )


            # ------------------------------------------------
            # text 不能为空
            # ------------------------------------------------

            if not str(
                record["text"]
            ).strip():

                raise ValueError(
                    (
                        "text 不能为空，"
                        f"line={line_number}"
                    )
                )


            records.append(
                record
            )


    # --------------------------------------------------------
    # 整个数据文件为空
    # --------------------------------------------------------

    if not records:

        raise RuntimeError(
            "知识数据为空。"
        )


    # --------------------------------------------------------
    # 检查 chunk_id 重复
    # --------------------------------------------------------

    chunk_ids = [
        record["chunk_id"]

        for record
        in records
    ]


    if len(
        chunk_ids
    ) != len(
        set(chunk_ids)
    ):

        raise ValueError(
            "知识数据中存在重复的 chunk_id。"
        )


    return records


# ============================================================
# 删除旧 Qdrant Local Storage
# ============================================================

def reset_qdrant_storage():
    print(f"Resetting Qdrant storage: {QDRANT_STORAGE_PATH}")

    QDRANT_STORAGE_PATH.mkdir(
        parents=True,
        exist_ok=True,
    )

    try:
        for item in QDRANT_STORAGE_PATH.iterdir():
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()

    except PermissionError as exc:
        raise RuntimeError(
            "无法清空 Qdrant 存储目录。"
            "请确认 FastAPI 或其他 QdrantClient 已关闭。"
        ) from exc

    remaining_items = list(QDRANT_STORAGE_PATH.iterdir())

    if remaining_items:
        raise RuntimeError(
            f"Qdrant 存储目录未完全清空: {remaining_items}"
        )

    print("Qdrant storage cleared.")


# ============================================================
# 创建全新的 Collection
# ============================================================

def create_collection(
    qdrant_client,
):

    print()
    print(
        "Creating Qdrant collection..."
    )


    # --------------------------------------------------------
    # 理论上整个数据库刚刚重新建立，
    # Collection 不应该已经存在。
    # --------------------------------------------------------

    if qdrant_client.collection_exists(
        collection_name=
            QDRANT_COLLECTION
    ):

        raise RuntimeError(
            (
                "新的 Qdrant Storage 中"
                "已经存在目标 Collection。"
            )
        )


    # --------------------------------------------------------
    # 创建 Collection
    # --------------------------------------------------------

    qdrant_client.create_collection(
        collection_name=
            QDRANT_COLLECTION,

        vectors_config=
            VectorParams(
                size=
                    EMBEDDING_DIMENSION,

                distance=
                    Distance.COSINE,
            ),
    )


    # --------------------------------------------------------
    # 新 Collection 必须为空
    # --------------------------------------------------------

    new_count = (
        qdrant_client
        .count(
            collection_name=
                QDRANT_COLLECTION,

            exact=
                True,
        )
        .count
    )


    print(
        "New collection points:",
        new_count,
    )


    if new_count != 0:

        raise RuntimeError(
            (
                "新建 Collection 不是空的，"
                f"points={new_count}"
            )
        )


# ============================================================
# 调用 Embedding API
# ============================================================

def embed_texts(
    embedding_client,
    texts,
):

    response = (
        embedding_client
        .embeddings
        .create(
            model=
                EMBEDDING_MODEL,

            input=
                texts,

            dimensions=
                EMBEDDING_DIMENSION,

            encoding_format=
                "float",
        )
    )


    # --------------------------------------------------------
    # 取出全部 Embedding
    # --------------------------------------------------------

    embeddings = [
        item.embedding

        for item
        in response.data
    ]


    # --------------------------------------------------------
    # 输入数量和 Embedding 数量必须一致
    # --------------------------------------------------------

    if len(
        embeddings
    ) != len(
        texts
    ):

        raise RuntimeError(
            (
                "Embedding 返回数量"
                "与输入文本数量不一致。 "
                f"texts={len(texts)}, "
                f"embeddings={len(embeddings)}"
            )
        )


    # --------------------------------------------------------
    # 检查 Embedding 维度
    # --------------------------------------------------------

    for index, embedding in enumerate(
        embeddings,
        start=1,
    ):

        if len(
            embedding
        ) != EMBEDDING_DIMENSION:

            raise RuntimeError(
                (
                    "Embedding 维度错误，"
                    f"index={index}，"
                    f"expected="
                    f"{EMBEDDING_DIMENSION}，"
                    f"actual="
                    f"{len(embedding)}"
                )
            )


    return embeddings


# ============================================================
# 构造 Qdrant Points
# ============================================================

def build_points(
    records,
    embeddings,
):

    # --------------------------------------------------------
    # 先保证两个列表长度一致
    # --------------------------------------------------------

    if len(
        records
    ) != len(
        embeddings
    ):

        raise RuntimeError(
            (
                "records 和 embeddings "
                "数量不一致。"
            )
        )


    points = []


    # --------------------------------------------------------
    # record 和 embedding 一一对应
    # --------------------------------------------------------

    for record, embedding in zip(
        records,
        embeddings,
    ):

        chunk_id = (
            record["chunk_id"]
        )


        point = PointStruct(

            # Qdrant 内部 Point ID
            id=
                create_point_id(
                    chunk_id
                ),


            # 1024 维向量
            vector=
                embedding,


            # 原始知识与 Metadata
            payload={

                "chunk_id":
                    chunk_id,


                "text":
                    record[
                        "text"
                    ],


                # source 不需要原始 JSONL
                # 每一行重复保存。
                #
                # 直接使用当前知识文件名。
                "source":
                    SOURCE_NAME,


                "equipment":
                    record[
                        "equipment"
                    ],


                "category":
                    record[
                        "category"
                    ],
            },
        )


        points.append(
            point
        )


    # --------------------------------------------------------
    # 最后检查 Point 数量
    # --------------------------------------------------------

    if len(
        points
    ) != len(
        records
    ):

        raise RuntimeError(
            (
                "Point 数量与知识记录"
                "数量不一致。"
            )
        )


    return points


# ============================================================
# 主函数
# ============================================================

def main():

    print()
    print(
        "=" * 60
    )

    print(
        "Industrial RAG Knowledge Index Builder"
    )

    print(
        "=" * 60
    )


    # ========================================================
    # 1. 加载原始知识
    # ========================================================

    print()
    print(
        "Loading knowledge records..."
    )


    records = (
        load_knowledge_records()
    )


    print(
        "Loaded records:",
        len(records),
    )


    print(
        "Source:",
        SOURCE_NAME,
    )


    # ========================================================
    # 2. 清理旧 Qdrant Storage
    # ========================================================
    #
    # 注意：
    #
    # 必须在创建 QdrantClient 之前删除目录。
    #
    # 否则 QdrantClient 已经打开数据库后，
    # Windows 下可能无法安全删除数据库文件。
    # ========================================================

    reset_qdrant_storage()


    # ========================================================
    # 3. 创建全新的 QdrantClient
    # ========================================================

    print()
    print(
        "Opening new Qdrant storage..."
    )


    qdrant_client = QdrantClient(
        path=
            str(
                QDRANT_STORAGE_PATH
            )
    )


    try:

        # ====================================================
        # 4. 创建空 Collection
        # ====================================================

        create_collection(
            qdrant_client
        )


        # ====================================================
        # 5. 创建 Embedding Client
        # ====================================================

        embedding_client = OpenAI(
            api_key=
                DASHSCOPE_API_KEY,

            base_url=
                DASHSCOPE_BASE_URL,
        )


        # ====================================================
        # 6. 准备所有文本
        # ====================================================

        texts = [
            record["text"]

            for record
            in records
        ]


        # ====================================================
        # 7. 批量生成 Embedding
        # ====================================================

        print()
        print(
            "Generating embeddings..."
        )


        embeddings = embed_texts(
            embedding_client=
                embedding_client,

            texts=
                texts,
        )


        print(
            "Generated embeddings:",
            len(embeddings),
        )


        # ====================================================
        # 8. 构造 Point
        # ====================================================

        print()
        print(
            "Building Qdrant points..."
        )


        points = build_points(
            records=
                records,

            embeddings=
                embeddings,
        )


        print(
            "Built points:",
            len(points),
        )


        # ====================================================
        # 9. 写入数据库
        # ====================================================

        print()
        print(
            "Writing points to Qdrant..."
        )


        qdrant_client.upsert(
            collection_name=
                QDRANT_COLLECTION,

            points=
                points,

            wait=
                True,
        )


        # ====================================================
        # 10. 查询最终数据库数量
        # ====================================================

        indexed_count = (
            qdrant_client
            .count(
                collection_name=
                    QDRANT_COLLECTION,

                exact=
                    True,
            )
            .count
        )


        # ====================================================
        # 11. 最终完整性验证
        # ====================================================

        if indexed_count != len(
            records
        ):

            raise RuntimeError(
                (
                    "Qdrant 索引数量"
                    "与原始知识数量不一致。 "
                    f"records={len(records)}, "
                    f"indexed={indexed_count}"
                )
            )


        # ====================================================
        # 12. 成功输出
        # ====================================================

        print()
        print(
            "=" * 60
        )

        print(
            "Index Build Completed"
        )

        print(
            "=" * 60
        )

        print()


        print(
            "Collection:",
            QDRANT_COLLECTION,
        )


        print(
            "Source:",
            SOURCE_NAME,
        )


        print(
            "Knowledge Records:",
            len(records),
        )


        print(
            "Generated Embeddings:",
            len(embeddings),
        )


        print(
            "Indexed Points:",
            indexed_count,
        )


        print(
            "Embedding Dimension:",
            EMBEDDING_DIMENSION,
        )


        print()


    finally:

        # ====================================================
        # 无论成功失败，关闭 Qdrant
        # ====================================================

        qdrant_client.close()


# ============================================================
# Python 模块入口
# ============================================================

if __name__ == "__main__":

    main()