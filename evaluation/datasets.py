# ============================================================
# Retrieval Evaluation Dataset
# ============================================================
#
# 每条数据包含：
#
# query
#   用户查询
#
# expected_chunk_id
#   人工标注的正确 Chunk
#
# type
#   查询类型，用于后续 Failure Analysis
#
# 类型说明：
#
# exact
#   精确故障码 / 标识符查询
#
# semantic
#   自然语言语义改写
#
# symptom
#   用户只描述故障现象
#
# maintenance
#   日常维护 / 检查问题
#
# confusing
#   与其他 Chunk 语义较接近，
#   用来测试 Retriever 的区分能力
# ============================================================


RETRIEVAL_TEST_CASES = [

    # ========================================================
    # 1. Exact Identifier
    # ========================================================

    {
        "query":
            "E-103 是什么故障？",

        "expected_chunk_id":
            "servo_e103",

        "type":
            "exact",
    },

    {
        "query":
            "E-101 报警应该检查什么？",

        "expected_chunk_id":
            "servo_e101",

        "type":
            "exact",
    },

    {
        "query":
            "PLC-ALARM-002 表示什么异常？",

        "expected_chunk_id":
            "plc_alarm_002",

        "type":
            "exact",
    },

    {
        "query":
            "AXIS-4 是什么故障？",

        "expected_chunk_id":
            "axis_4",

        "type":
            "exact",
    },


    # ========================================================
    # 2. Semantic Paraphrase
    # ========================================================

    {
        "query":
            "控制柜太热应该排查什么？",

        "expected_chunk_id":
            "cabinet_cooling",

        "type":
            "semantic",
    },

    {
        "query":
            "控制柜完全启动不了应该先检查哪里？",

        "expected_chunk_id":
            "cabinet_power",

        "type":
            "semantic",
    },

    {
        "query":
            "机器人突然和控制系统连不上了应该检查什么？",

        "expected_chunk_id":
            "network_disconnect",

        "type":
            "semantic",
    },

    {
        "query":
            "PLC 和工业网络通信异常应该排查哪些地方？",

        "expected_chunk_id":
            "plc_alarm_003",

        "type":
            "semantic",
    },

    {
        "query":
            "伺服驱动器提示编码器通信异常应该检查什么？",

        "expected_chunk_id":
            "servo_e104",

        "type":
            "semantic",
    },

    {
        "query":
            "第六轴发生故障应该检查哪些部件？",

        "expected_chunk_id":
            "axis_6",

        "type":
            "semantic",
    },


    # ========================================================
    # 3. Symptom Query
    # ========================================================

    {
        "query":
            "机器人运行时出现异常响声应该检查哪些部件？",

        "expected_chunk_id":
            "abnormal_noise",

        "type":
            "symptom",
    },

    {
        "query":
            "机器人润滑不好会造成哪些问题？",

        "expected_chunk_id":
            "lubrication_insufficient",

        "type":
            "symptom",
    },

    {
        "query":
            "急停按钮已经复位，但机器人还是无法解除急停怎么办？",

        "expected_chunk_id":
            "cabinet_emergency_stop",

        "type":
            "symptom",
    },


    # ========================================================
    # 4. Maintenance Query
    # ========================================================

    {
        "query":
            "机器人每天启动之前需要检查哪些项目？",

        "expected_chunk_id":
            "daily_inspection",

        "type":
            "maintenance",
    },

    {
        "query":
            "机器人日常保养时润滑部分需要检查哪些内容？",

        "expected_chunk_id":
            "lubrication_check",

        "type":
            "maintenance",
    },


    # ========================================================
    # 5. Confusing / Similar Knowledge
    # ========================================================

    {
        "query":
            "伺服驱动器发生过载时应该检查哪些方面？",

        "expected_chunk_id":
            "servo_e102",

        "type":
            "confusing",
    },

    {
        "query":
            "机器人运行负荷超过额定范围应该怎么处理？",

        "expected_chunk_id":
            "overload_operation",

        "type":
            "confusing",
    },

]


# ============================================================
# Answerability Evaluation Dataset
# ============================================================
#
# expected_status:
#
# ACCEPT
#   当前知识库中有足够证据回答
#
# REJECT
#   当前知识库中没有足够证据回答
# ============================================================


ANSWERABILITY_TEST_CASES = [

    # ========================================================
    # Answerable
    # ========================================================

    {
        "query":
            "E-103 是什么故障？",

        "expected_status":
            "ACCEPT",
    },

    {
        "query":
            "控制柜太热应该检查什么？",

        "expected_status":
            "ACCEPT",
    },

    {
        "query":
            "机器人和控制系统连不上了应该排查什么？",

        "expected_status":
            "ACCEPT",
    },

    {
        "query":
            "机器人运行时出现异常噪声应该检查哪些地方？",

        "expected_status":
            "ACCEPT",
    },

    {
        "query":
            "机器人日常维护时润滑部分要检查什么？",

        "expected_status":
            "ACCEPT",
    },

    {
        "query":
            "PLC-ALARM-002 表示什么问题？",

        "expected_status":
            "ACCEPT",
    },

    {
        "query":
            "急停复位以后仍然无法解除急停应该怎么办？",

        "expected_status":
            "ACCEPT",
    },

    {
        "query":
            "机器人运行负荷超过额定范围应该怎么处理？",

        "expected_status":
            "ACCEPT",
    },


    # ========================================================
    # Unanswerable - Topic Related but Evidence Insufficient
    # ========================================================

    {
        "query":
            "ABB IRB6700 推荐使用哪个品牌的润滑油？",

        "expected_status":
            "REJECT",
    },

    {
        "query":
            "ABB IRB6700 更换减速器大概需要多少钱？",

        "expected_status":
            "REJECT",
    },

    {
        "query":
            "PLC-ALARM-003 一年大约会发生多少次？",

        "expected_status":
            "REJECT",
    },

    {
        "query":
            "E-103 故障通常需要多久才能维修完成？",

        "expected_status":
            "REJECT",
    },

    {
        "query":
            "ABB IRB6700 的控制柜冷却风扇是什么品牌？",

        "expected_status":
            "REJECT",
    },

]