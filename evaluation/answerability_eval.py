from app.bootstrap import create_rag_pipeline

from evaluation.datasets import (
    ANSWERABILITY_TEST_CASES,
)


def main():

    rag_pipeline, qdrant_client = (
        create_rag_pipeline()
    )

    try:

        correct = 0

        false_accept = 0

        false_reject = 0


        for case in ANSWERABILITY_TEST_CASES:

            query = case["query"]

            expected_status = (
                case["expected_status"]
            )


            result = (
                rag_pipeline.answer(
                    query
                )
            )


            actual_status = (
                result["status"]
            )


            is_correct = (
                actual_status
                == expected_status
            )


            if is_correct:

                correct += 1


            # =================================================
            # False Accept
            # =================================================

            if (
                expected_status == "REJECT"
                and
                actual_status == "ACCEPT"
            ):

                false_accept += 1


            # =================================================
            # False Reject
            # =================================================

            if (
                expected_status == "ACCEPT"
                and
                actual_status == "REJECT"
            ):

                false_reject += 1


            # =================================================
            # 单条结果
            # =================================================

            print(
                query
            )

            print(
                "expected:",
                expected_status,
            )

            print(
                "actual:",
                actual_status,
            )

            print(
                "top1_score:",
                f"{result['top1_score']:.4f}",
            )

            print(
                "-" * 60
            )


        total = len(
            ANSWERABILITY_TEST_CASES
        )


        print()

        print(
            "========== Answerability Evaluation =========="
        )

        print()


        print(
            "Total:",
            total,
        )

        print(
            "Correct:",
            correct,
        )

        print(
            "Accuracy:",
            f"{correct / total:.2%}",
        )

        print(
            "False Accept:",
            false_accept,
        )

        print(
            "False Reject:",
            false_reject,
        )


    finally:

        qdrant_client.close()


if __name__ == "__main__":
    main()