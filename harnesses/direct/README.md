# direct harness

单轮指令遵循：原始 prompt → 模型回答 → 环境判分。当前配套 IFEval，支持 collect/train/evaluate 和本地模型。复用公共 token 渲染、采样和 GRPO 数据记录代码，不调用 arl_agent，不执行 agent 工具循环。
