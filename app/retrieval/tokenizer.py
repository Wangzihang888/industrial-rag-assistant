import re

import jieba


def industrial_tokenize(text):
    """
    面向工业设备文档的简单分词函数。

    目标：
    1. 尽量保留 E-103、AXIS-4、PLC-ALARM-003
       这类工业故障码作为完整 token。
    2. 对剩余中文文本使用 jieba 分词。
    """

    codes = re.findall(
        r"[A-Za-z]+(?:-[A-Za-z0-9]+)+",
        text,
    )

    remaining_text = text

    for code in codes:
        remaining_text = remaining_text.replace(
            code,
            " ",
        )

    chinese_tokens = list(
        jieba.cut(
            remaining_text
        )
    )

    chinese_tokens = [
        token.strip()
        for token in chinese_tokens
        if token.strip()
        and not re.fullmatch(
            r"\W+",
            token,
        )
    ]

    return codes + chinese_tokens