# arl_agent

ARL 的 CUDA 算子优化 / KernelBench agent harness。仅配套 coding.cuda；可通过 CudaTasks 选择算子题库。负责工具循环、上下文处理及原生 GRPO token 轨迹。SWE、ProgramBench、IFEval 不允许选择本 harness。

公共 token 渲染和采样代码可由不同 harness 复用，复用公共组件不意味着执行 arl_agent。
