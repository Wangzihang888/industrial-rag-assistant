class RAGServiceError(Exception):
    """
    RAG 服务执行过程中发生的异常。
    """

    def __init__(
        self,
        message="RAG 服务执行失败",
    ):
        self.message = message

        super().__init__(message)