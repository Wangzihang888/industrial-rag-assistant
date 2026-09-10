from app.config import (
    LLM_MODEL,
)

from app.generation.context import (
    build_context,
)

from app.exceptions import (
    RAGServiceError,
)


class QwenGenerator:

    def __init__(
        self,
        llm_client,
    ):
        self.llm_client = (
            llm_client
        )


    def generate(
        self,
        query,
        reranked_results,
    ):

        # ====================================================
        # 1. 构造 Context
        # ====================================================

        context = build_context(
            reranked_results
        )


        # ====================================================
        # 2. System Prompt
        # ====================================================

        system_prompt = """
你是一个工业设备运维知识库助手。

你必须严格遵守以下规则：

1. 只能根据提供的 Context 回答。
2. 不允许使用 Context 之外的信息补充答案。
3. 如果 Context 中没有明确证据支持某个结论，
   不要猜测，也不要编造。
4. 回答应简洁、专业、直接。
5. 关键结论必须引用对应来源，
   格式使用 [Source 1]、[Source 2]。
6. 不允许引用不存在的 Source。
7. 如果多个 Source 提供互补信息，
   可以同时引用多个 Source。
"""


        # ====================================================
        # 3. User Prompt
        # ====================================================

        user_prompt = f"""
请根据下面提供的 Context 回答用户问题。

User Query:
{query}

Context:
{context}

请给出最终答案，并在相关结论后标注引用来源。
"""


        # ====================================================
        # 4. 调用 LLM
        # ====================================================

        try:

            response = (
                self.llm_client
                .chat
                .completions
                .create(
                    model=
                        LLM_MODEL,

                    messages=[
                        {
                            "role":
                                "system",

                            "content":
                                system_prompt,
                        },

                        {
                            "role":
                                "user",

                            "content":
                                user_prompt,
                        },
                    ],

                    temperature=0,
                )
            )


            answer = (
                response
                .choices[0]
                .message
                .content
            )


        except Exception as exc:

            raise RAGServiceError(
                "LLM API 调用失败"
            ) from exc


        # ====================================================
        # 5. 检查模型返回内容
        # ====================================================

        if not answer:

            raise RAGServiceError(
                "LLM 返回了空答案"
            )


        return answer