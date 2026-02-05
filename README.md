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


# [大模型裁剪系列]
1. DATA-FREE PRUNING OF SELF-ATTENTION LAYERS IN LLMS
   * 论文发现随着seq的增加，attention的耗时占比要大于FFN模块，于是提出了指裁剪attention的data-free方案
   * 方法是提出gate-norm，计算q-k之间的weight的耦合度，不需要校准集和微调，gate-norm=WqWk^T
   * 效果，通过论文的方法，可以实现1.3X的推理提速。
2. ShortGPT: Layers in Large Language Models are More Redundant Than You Expect-百川智能
   * 论文发现相邻层之间的隐藏状态具有极高的相似性，意味着这些层对输入信息的转换非常小，小到可以移除
   * 方法是计算每一层的输入和输出的余弦相似度，如果相似度高则重要性低，然后排序之后，移除N层。
3. What Matters in Transformers? Not All Attention is Needed
   * 论文发现注意力机制是Transformer的核心，但是大量的注意力层实际上是多余的，可以在不显著影响模型性能的情况下直接删除
   * 相似度的计算则是采用余弦相似度，相似度越高则越不重要，可以删除。
   * 由于残差结构的存在，论文提出了三种删除方法，整块删除、MLP删除和Attention删除，前面两个删除会导致性能下降，但是Attention的删除则可以保留极高的准确度。
   * 在llama-2-13B上测试，删除8层有1.13X提速且基本不掉点，删除20层有1.40X提速且掉点在1%以内。
4. SparseGPT: Massive Language Models Can be Accurately Pruned in One-Shot
   * 论文把全局剪枝问题分解为对每一层权重矩阵进行局部最优化的大规模稀疏回归问题
   * 步骤1：采用逐层剪枝策略，对于每一个线性层，目标是找到一个新的稀疏重矩阵，使得该层输出误差最小化，min(||Wx -W'x||2
   * 步骤2：基于Hessian矩阵精准修正，为了在删除权重的时候不丢失精度，通过H=XX^T的方式动态调整该行的剩余未被裁剪的权重
   * 步骤3：为了提高计算效率，算法以权重块为单位进行处理，而不是逐个权重处理
5. Wanda: A SIMPLE AND EFFECTIVE PRUNING APPROACH FOR LARGE LANGUAGE MODELS-博世
   * 针对剪枝需要Hessian矩阵调整权重的复杂，提出一个极简、高效且不需要更新权重的剪枝方案。
   * 新的权重重要性度量，论文提出，针对大模型的情况，如果一个权重的输入非常大，就算权重本身很小，它对输出的影响也很大，所以采用s=|W|*||X||2的方式进行度量。计算出s之后，进行排序和裁剪就行。当然也可以针对nv的2：4稀疏化进行剪枝。

# [VLA经典]
1. OpenVLA

# [字节的GR系列]
1. GR-3技术报告
   * VLA的架构采用qwen2.5-vl-3b的VLM模块，加速1B参数的动作扩散模型(DiT),这里的扩散模块采用流匹配来预测动作块
   *  DiT 模块的注意力层和前馈网络（FFN）中引入了 RMSNorm，这显著提高了训练稳定性和指令遵循能力。
   *  训练分为3个阶段： 1）视觉-语音联合训练，2）机器人轨迹模仿学习，3）人类轨迹微调
   *  KV Cache注入：架构中有三个箭头从VLM->DiT，表示VLM传递给DiT的是KV cache，论文从效率出发，指选取了VLM的后半部分层的KV cache，这里的KV cache象征着“语义+空间+状态”的高维表征从VLM传递到DiT。
   *  任务状态的有效性：动作模块引入辅助任务的任务状态（包括进行中、已完成和无效指令）能有效帮助这个 4B 参数的 VLA 模型更好地将视觉场景与语言指令对齐，确保机器人真的在按照指令行事，而不是盲目地重复动作。
2. GR-RL: Going Dexterous and Precise for Long-Horizon Robotic Manipulation
   * 架构和动作模块和GR-3保持一致
   * 论文引入了在线强化学习，在穿鞋带任务中，GR-RL 从基础模型的 45.7% 成功率大幅提升至 83.3%。
   * 补充图片，图片有完整的结构信息，包括kv-cache的传递，强化学习等
   * 这里的强化学习没有明白，后续补充知识点

# [google的RT系列]
1. RT-1:Robotics Transformer for Real-World Control at Scale
   * 架构上：1）视觉和语言融合模块，时序6帧图像和任务内容输入到FiLM模块，输出6* 81个token，2)视觉token裁剪，TokenLearner模块，通过注意力机制过滤token，输出6* 8个token，减少计算负担，3)动作预测使用decoder-only的transformer模块，输出预测离散化的动作token，4）离散化动作表示，对动作detoken，采用分箱处理，输出每个关节的离散量。
   * 推理：推理速度可以达到3Hz，使用TPUv4芯片，主要的加速技术包括视觉token压缩和FiLM使用efficientNet的变体
   * 任务定义：论文采用自然语言指令为准，指标格式为动词+名称的形式，如打开抽屉
   * 实验证明：RT-1的未见任务的泛化性强，异构数据的吸收能力强和长程任务的有效性。
3. RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control
   * 架构上：提出了VLA的架构，就是在PaLI-X的VLM的基础上，增加了动作的标记位。
   * 动作标记化：论文将机器人的连续动作空间映射到了模型的文本标记空间（Vocabulary）。做法是1）动作空间：包括 6 自由度的末端执行器位姿（x, y, z, roll, pitch, yaw）、夹持器的张合程度以及一个表示任务是否终止的特殊命令。2）离散化：将这些连续的数值划分为 256 个等分区间（bins），每个区间对应一个整数。3）映射：将这 256 个区间与语言模型中已有的文本 Token 关联。4）输出格式：模型输出的是一串代表动作的 Token，例如 1 128 91 241 5 101 127，然后通过反标记化（De-tokenization）转换为机器人硬件可执行的物理指令。
   * 训练策略：采用了共微调（co-fine-tuning)的方式，在每一个训练批次中，同时包含互联网规模的视觉问答数据和机器人抓取轨迹数据。这样做可以防止模型在学习机器人技能时出现“灾难性遗忘”。
5. RT-H
6. RT-X

# [Physical Intelligence系列]
1. pi0：
2. 


# [claude code使用技巧]
1.命令行启动：claude
2.定制claude的工作环境：/init ,会创建claude.md文件，里面可以写一些架构的东西，如编程语音
3.构建高效的提示工程：（坏的）添加一个日历组件，（好的）查看现有组件如何集成到仪表盘的，使用react创建一个日期选择器组件
4.使用思考引导词："think","think hard", "think harder", "ultrathink"
5.利用并行处理：使用子代理，提示词写法，你是一个bug修复的子代理，你需要完成如下的功能，修改当前运行的bug。
6.标准工作流程：1）探索阶段，如读取整个文件，理解里面的代码逻辑；2）规划阶段，要求claude制订解决问题的计划，使用think;3）编码阶段，让claude编码实现；4）提交阶段：让claude提交结果并创建git commit
