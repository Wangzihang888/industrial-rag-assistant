from app.config import (
    EVIDENCE_THRESHOLD,
)


class EvidenceGate:

    def __init__(
        self,
        threshold=EVIDENCE_THRESHOLD,
    ):
        """
        Evidence Gate

        threshold:
            判断证据是否充分的阈值。
        """

        self.threshold = threshold


    def check(
        self,
        reranked_results,
    ):
        """
        判断当前检索证据是否足够。

        返回：
            {
                "accepted": True / False,
                "top1_score": float,
                "threshold": float,
            }
        """

        if not reranked_results:

            return {
                "accepted": False,
                "top1_score": 0.0,
                "threshold": self.threshold,
            }


        top1_score = (
            reranked_results[0][
                "rerank_score"
            ]
        )


        accepted = (
            top1_score
            >= self.threshold
        )


        return {
            "accepted": accepted,
            "top1_score": float(
                top1_score
            ),
            "threshold": float(
                self.threshold
            ),
        }