def build_context(
    reranked_results,
):
    """
    将 Reranker 返回的 Top-K 结果
    格式化成提供给 LLM 的 Context。
    """

    context_parts = []

    for index, item in enumerate(
        reranked_results,
        start=1,
    ):

        context_part = (
            f"[Source {index}]\n"
            f"chunk_id: "
            f"{item['chunk_id']}\n"
            f"source: "
            f"{item.get('source', 'unknown')}\n"
            f"equipment: "
            f"{item.get('equipment', 'unknown')}\n"
            f"category: "
            f"{item.get('category', 'unknown')}\n"
            f"text: "
            f"{item['text']}"
        )

        context_parts.append(
            context_part
        )

    context = "\n\n".join(
        context_parts
    )

    return context