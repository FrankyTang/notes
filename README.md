# notes
some knowledges that are importent or  easy to forget, including not only AI.

# 视觉token裁剪
1. 小鹏-fastDriveVLA
2. 腾讯-VScan https://hub.baai.ac.cn/view/47057


# [nvidia]Alpamayo-R1
涉及到的推理加速有如下方面：
1. 视觉编码模块：使用flex的视频tokenizer,使用3平面特征移除重复的图像编码。
2. 语言模型规模：使用qwen3-0.5B的模型参数
3. decode阶段：精简输出，token最大控制在40
4. 动作解码阶段：使用flow matching 轨迹解码器
5. 芯片选型：使用RTX6000pro(FP16精度达到503TFPLOPS。

时延拆解（99ms)：视觉编码(3.43ms)、prefilling阶段(16.54ms)、decode阶段(70ms，约40个token)、轨迹生成(8.75ms)
如果不使用 Flow Matching 而使用自回归方式生成轨迹，总延迟会飙升至 312ms，这将无法满足实时驾驶的需求。

[] 为什么没有使用视觉token裁剪？
论文把视觉token裁剪定义为一个补充途径，引用了 SparseVILA (Khaki et al., 2025)。这项技术的特点是：
动态识别： 在推理过程中实时识别并移除冗余的视觉 Token。
无需重训： 它可以直接应用于已经训练好的模型（Post-training pruning），从而降低推理成本。

[] 涉及到其他推理加速的手段不？
论文在结论和讨论中暗示，对于更大规模（如 3B 或 7B）的模型实现实时化，除了 Token 压缩外，**量化和模型剪枝（Pruning）**是后续扩展时的可选手段（例如引用了 SparseVILA），但在当前的 AR1 核心实验中，99ms 是在没有依赖激进量化的前提下通过架构优化实现的。

[] 为什么不使用投机采样？
大模型反馈：投机采样在大型语言模型社区中很流行，但对于自动驾驶这种实时控制任务，开发者往往更倾向于使用 Flow Matching 这种计算时间非常恒定（Deterministic）的算法，而不是投机采样这种加速比随场景复杂度波动的算法。

[] 三阶段如何训练的？
1. 动作能力注入：冻结VLM大部分参数，训练视觉编码适配器和action—head，让模型理解动作特征
2. 监督微调：最小化推理文本和预测轨迹的联合交叉熵损失。通过这种方式，模型学会了将视觉观察（如“前方有行人”）与文字逻辑（如“我需要减速”）以及物理动作（如“刹车轨迹”）进行强关联。
3. 强化学习后训练：采用 GRPO（Group Relative Policy Optimization，一种不需要独立评论员模型的强化学习算法，也用于 DeepSeek-R1）
