# MemRL 源码交付与部署运行说明

更新日期：2026-10-16。适用于双方共用现有 coding 和远端服务器、已有模型及 Python 环境的情形。不包含服务器登录配置；以下 coding 命令均在 CodingUbuntu 的 Linux 终端执行。

2026-10-16执行框架变更：SWE Solver已改为mini-swe-agent的真实`DefaultAgent`循环，使用bash代码块执行命令，替换原来自写的JSON action/final循环。主任务、局部分支、冻结测试均走同一个适配层，E/U/R仍使用HF。本次只修改源码和配置、做离线检查，不调用模型、不启动训练；下文2026-10-15的模型验收不能视为新执行器已通过实机做题。已有`releases/current`交付包也不代表本次新源码，需重新打包后再交付。

同日并行修正：默认配置已启用两条主轨迹并行，每个Solver固定承担一个replica；E/U/R调用和轨迹预处理串行访问共享HF模型及tokenizer。环境服务改为按执行/容器加锁，耗时工具与判题不再持有全局锁；启动入口会拒绝复用旧并发协议的环境服务。只做离线验收，未启动30+12实验。

本日后续变更：用户明确选择方法三作为默认。当前新运行使用HF缓存采样＋单步策略梯度；0.03只标记有限概率差异，不再阻止默认PG启动或更新。下文既有验收记录属于变更前的严格方案，不能作为方法三学习效果证明。

## 1. 交付内容与当前部署

交付两个文件：`MemRL_source_share_20261015.tar.gz` 和本文。包内 `使用说明.md` 与本文内容一致；`MANIFEST.sha256.json` 用于检查文件是否变化。源码、完整训练配置、小型题目清单、启动脚本和测试都在包内。大模型、SFT权重、完整数据集、Docker镜像和Python环境使用服务器已有文件，不装入压缩包。

| 内容 | coding绝对路径 |
|---|---|
| 唯一源码目录 | `/home/ubuntu/coding/MemRL/02_源码与脚本` |
| 源码压缩包 | `/home/ubuntu/coding/MemRL/02_源码与脚本/releases/current/MemRL_source_share_20261015.tar.gz` |
| 单独交付文档 | `/home/ubuntu/coding/MemRL/02_源码与脚本/releases/current/MemRL_部署与运行说明_20261015.md` |
| 开发源码 | `/home/ubuntu/coding/MemRL/02_源码与脚本` |
| 包内默认完整配置 | `configs/coding_hf.json` |
| 新运行输出父目录 | `/home/ubuntu/coding/MemRL/03_实验结果/runs/memrl` |

| 组件 | 所在位置与调用方式 | 当前职责 |
|---|---|---|
| Solver、Pair Judge | coding GPU0；`http://127.0.0.1:8000/v1`；冻结Qwen3.5-4B | 主任务、局部分支做题和结果比较 |
| E/U/R | coding GPU1；HF训练进程直接加载，没有HTTP端口 | E/U采样、R候选评分、概率重算及LoRA更新 |
| Embedding | 远端GPU1；coding通过`http://127.0.0.1:8002/v1`调用 | 4096维记忆向量 |
| SWE环境和官方判题 | coding Docker；`http://127.0.0.1:18119` | 隔离工具执行、补丁归档及SWE-bench判题 |

记忆模块已统一使用HF。Solver和Embedding当前仍用vLLM服务，它们不参与E/U/R反向传播。`local_solver_endpoints`中的18281可作为额外冻结Solver端点运行局部分支，不是E/U/R的vLLM采样或训练后端。LoRA发布RPC和HF/vLLM概率修正不用于当前Runtime。

### SWE Solver的执行方式

2026-10-16提示词校正：`domains/reasoningbank_prompts.py`逐字保留此前ReasoningBank运行的系统、题目、工具观察和格式反馈模板；无记忆时系统提示词原样使用，有记忆时按ReasoningBank追加原记忆使用说明和R选出的记忆块。记忆块仍由MemRL渲染，E/U/R构建和筛选方式不变。harness revision升级，须新建运行目录，不恢复旧提示词协议的RL checkpoint。双Solver的有/无记忆首响应共4次API诊断均产生合法单bash块，但不代表完整任务正确率；原简化提示词组未进行RL更新，日志保留。

`domains/swe.py`选择`MiniSWEAdapter`，由独立Python进程中的mini-swe-agent管理任务循环、格式反馈和提交。父进程只桥接现有模型API、工具接口和预算，不再解析Solver的JSON action/final。模型请求仍经过原来的`ChatModel`，保留`api_requests.jsonl`、`api_responses.jsonl`及token计数；Docker仍使用原来的固定镜像、reset校验、工具输出归档和官方判题。mini不会再创建第二个题目容器。

默认配置：

```json
"consumer": {
  "runner": "mini_swe_agent",
  "mini_swe": {
    "python": "/home/ubuntu/coding/venvs/swebench/bin/python",
    "max_consecutive_format_errors": 0
  }
}
```

这里仅展示新增配置项，其余模型地址和profile字段仍需保留。`python`指向已有mini-swe-agent 2.x环境，与HF训练Python分开，不需要给训练环境安装mini或LiteLLM；更换机器时修改此路径。`0`表示格式错误后可以继续尝试，但始终受总调用次数、token和wall预算限制，不会无限重试。Solver的JSON输出约束已移除；Judge和U的JSON协议、E的`e-text-v1`不变。

Solver每次输出一个命令，例如：

````text
检查相关源码。
```bash
sed -n '1,120p' source.py
```
````

完成时执行`echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`。只有命令退出码为0且输出首行为该标记才视为提交；真正补丁仍由环境`finalize`提取，提交不代表官方判题成功。空正文、无命令或多个代码块进入格式反馈；输出因token上限截断时不执行其中的命令。基础设施错误仍明确报错，不能冒充做题失败。

每个任务开始时R只选择一次记忆，注入系统提示词；内部工具步骤不调用E/U/R。局部分支只执行当前短预算内的步骤，然后交给Judge，不调用E/U或完整官方判题。主任务和冻结测试仍按原流程判题。

每次执行额外保存`<output_dir>/solver_trajectories/<execution_id的hash>.traj.json`及对应stderr日志。原生轨迹包含mini版本、输入消息、命令、工具结果、调用计数及预算；`info.memrl.execution_id`可反查执行ID，正常结束后补充实际补丁和状态hash。E仍使用公开结构化事件及现有32K预处理，不接收私有测试日志。

仅检查mini依赖、无需访问模型或Docker的命令：

```bash
cd /home/ubuntu/coding/MemRL/02_源码与脚本
/home/ubuntu/coding/venvs/swebench/bin/python \
  src/eur_memory_rl/domains/mini_swe_worker.py --check
```

正常启动仍使用`bash scripts/memrl.sh`，不新增训练入口。启动准备阶段会把SWE配置规范为新执行器，记录新的`harness_revision`和配置指纹；Runtime在加载GPU权重前检查mini环境。不要直接恢复旧工具循环暂停中的进程，也不要把旧批次轨迹当作新协议下采样的训练样本；切换后使用新的运行目录。实际并行性能和做题效果仍需实测，不能由离线检查推断加速倍数。

### 双Solver与环境服务的并行边界

当前默认配置的相关字段如下；其他配置项仍需保留：

```json
{
  "consumer": {"base_url": "http://127.0.0.1:8000/v1"},
  "local_solver_endpoints": ["http://127.0.0.1:18281/v1"],
  "global": {"replicas": 2, "main_parallel_workers": 2}
}
```

`consumer.base_url`承担replica-0，第二个端点承担replica-1，两条任务流有各自的HTTP客户端、容器和记忆库。同一任务流内仍按题目顺序执行；一条流等待HF时，另一Solver可以继续做题。E/U/R的采样、adapter切换、索引和共享tokenizer操作由同一把锁保护，并在工作线程内显式进入注意力上下文。主轨迹全部返回后才运行局部阶段，整组局部采样结束后才更新参数。失败的主轨迹组整体排除，不把部分完成的样本交给更新。

局部分支继续复用这两个端点，主任务与局部任务不会争抢同一个Solver客户端。所有端点统一记录执行、判题及超时后的有效奖励规则。`solver_dispatch.jsonl`逐条记录执行ID、worker编号、开始/完成/失败和耗时；`solver_parallelism.json`记录端点与更新边界。最终12题冻结测试目前仍按原逻辑串行，本次没有改为并行。

环境服务只在查询/修改会话和幂等缓存时持有短暂全局锁；同一执行的命令、finalize、close和判题仍串行，不同执行可以并行。相同请求ID的并发重试只执行一次；同ID不同内容会被拒绝。WebArena仍保持共享后端独占要求。

`/health`现在返回`concurrency_protocol: per-execution-v1`。当`main_parallel_workers > 1`时，即使环境配置指纹相同，启动入口也拒绝旧版服务，不会自动停止该端口的现有进程。端口空闲时沿用`--start-environment`启动本轮新服务。部署审查时18119空闲，两个Solver接入端口及Embedding转发端口均在监听；仅端口监听不代表模型调用验收通过。

如需串行对照，将新配置的`global.main_parallel_workers`设为1，并创建新运行目录。并行配置必须使用足够且不重复的端点，目前只开放给SWE；这些条件在加载GPU权重前检查。历史30+12运行目录及其配置未修改，仍需另行创建本轮配置；五组首题no-memory轨迹复用尚未在本次并行修正中扩展。

这里所有`127.0.0.1`均指命令所在机器。coding的8002是转发端口，远端Embedding实际监听18082；远端隧道把二者接通。不是把Embedding部署到coding CPU。

## 2. 最短使用流程

### 五组实验逐组启动与恢复（2026-10-16）

新配置 `configs/20261016-mini双Solver-EU-SFT-逐组.json` 固定同一份 20 主任务＋10 探针＋12 测试清单、五组首题 no-memory 文件及校验值。两个 replica 复用各组同一首题公开轨迹，但分别重新采样 E/U、维护独立空记忆库。历史 Solver 成本保留用于原奖励核算，新调用数单独记录为零；历史轨迹不是新执行器的 on-policy Solver 样本。

首次通过 `--sft-checkpoint` 指定 U SFT 的 `iteration-000088`（含已有 E SFT 和新 U SFT），使用新 RL 优化器。命令参数 `--max-updates N` 是累计更新边界，不是本次增加 N 次：首次 `--train --full-suite --max-updates 1`，第二次在同一 `--prepared-root` 下加 `--resume-checkpoint <iteration-000001目录> --max-updates 2`，以此至 5。始终通过 `bash scripts/memrl.sh` 启动，不改既有配置、数据清单或源码。每次仍加 `--start-environment`，只清理本次启动的环境桥。

前四组完成后以 `paused_after_update` 正常退出，不执行冻结测试。恢复会加载完整 LoRA、优化器、RNG、数据游标；组间记忆库按独立任务流设计清空。第 5 组更新后才用最终参数重建只读记忆库、统一测试 12 题；没有中间测试。组末 checkpoint 使用原子发布并有 SHA256，不能把 `--sft-checkpoint` 权重导入当成第二至第五组的 RL 恢复。当前只保护已完成组；组内失败没有任意步骤恢复承诺。源码/依赖检查和长输入预检仍在每次新进程启动时运行。

源码已在服务器上时，直接使用下述路径。若另行解压，可解压至新的目录；脚本根据自身位置寻找源码和包内配置，不要求覆盖开发目录。

```bash
cd /home/ubuntu/coding/MemRL/02_源码与脚本

# 查看参数。包装脚本自动使用现有训练Python，无需激活环境。
bash scripts/memrl.sh --help

# 准备新运行目录：只读模型路径和清单，不加载GPU模型。
# 将下面目录名改成自己的名称；目录必须尚不存在。
RUN_DIR=/home/ubuntu/coding/MemRL/03_实验结果/runs/memrl/my-hf-run-001
bash scripts/memrl.sh --prepare-only --root "$RUN_DIR"

# 临时启动本轮环境服务，实际请求模型API并检查Docker工具链。
# 结束时关闭本命令启动的环境服务；Solver和Embedding保持运行。
bash scripts/memrl.sh --prepared-root "$RUN_DIR" \
  --check --start-environment --environment-smoke \
  --report "$RUN_DIR/check.json"

# 正式训练：沿用仅准备过的目录，自动启动本轮环境服务。
# 会先做模型审计及32K反向预检，再做题和更新参数。
bash scripts/memrl.sh --prepared-root "$RUN_DIR" --train --start-environment
```

若只想验证启动能力，最后一条改为：

```bash
bash scripts/memrl.sh --prepared-root "$RUN_DIR" --startup-only --start-environment
```

启动验收不会执行真实Solver任务流或optimizer.step，但会计算梯度、创建初始化checkpoint。验收完成后，该目录已运行过，不能再当成全新的训练目录使用；正式训练请创建另一个新目录。

也可省略准备步骤，一条命令自动生成新目录：

```bash
# 默认仅启动验收，不做题、不更新。
bash scripts/memrl.sh

# 明确执行正式训练；运行目录自动生成并打印。
bash scripts/memrl.sh --train --start-environment
```

长任务建议在既有tmux会话里执行。运行中另一个终端查看`tail -f /本轮目录/launcher.log`。退出码非0时查看`summary.json`和`traceback.log`；若模型尚未加载，优先查看`service_checks.json`。不要把没有`completed_updates`的启动通过当作已经训练完成。

## 3. 启动参数与运行目录

`bash scripts/memrl.sh`调用固定Python和`start_memrl.py`；会清除继承的`CUDA_VISIBLE_DEVICES`，按配置中的物理GPU编号使用GPU1。直接调用Python时，需自行确保没有错误的GPU编号映射。

| 参数 | 行为 |
|---|---|
| `--check` | 文件、清单hash、依赖版本、GPU、Docker、镜像、实际Solver/Embedding请求、环境health；不加载训练模型 |
| `--environment-smoke` | 仅配合`--check`，创建一个临时题目容器，验证reset/step/finalize/close；不执行官方判题 |
| `--prepare-only` | 生成独立完整配置、复制小清单；不开始训练 |
| `--startup-only` | 模型审计、真实SFT权重加载后审计、32K+1反向、R16候选和E/U短生成；也是默认模式 |
| `--train` | 通过前置检查后执行当前清单的训练与冻结测试 |
| `--start-environment` | 本轮按需启动18119环境服务，结束时只停止自己启动的服务；已有服务须配置指纹匹配 |
| `--serve-environment` | 使用`--prepared-root`在前台单独运行环境服务 |
| `--root PATH` | 指定尚不存在的新运行目录 |
| `--prepared-root PATH` | 使用只准备过的目录；不能与`--root`或`--config`同时使用 |
| `--config PATH` | 完整配置，默认包内`configs/coding_hf.json`；相对资源路径以该配置所在目录为基准 |
| `--sft-checkpoint PATH` | 默认本文指定的文本E SFT初始化权重 |
| `--source-dir PATH` | 明确指定源码目录，供包外独立脚本使用 |
| `--report PATH` | 保存`--check`的JSON报告 |

例如明确指定配置和权重：

```bash
bash scripts/memrl.sh \
  --config "$PWD/configs/coding_hf.json" \
  --sft-checkpoint /home/ubuntu/coding/MemRL/03_实验结果/checkpoints/E文本SFT30_RL初始化权重 \
  --root /home/ubuntu/coding/MemRL/03_实验结果/runs/memrl/my-hf-run-002 \
  --train --start-environment
```

要改题目、预算或学习率，先复制并修改完整配置，再创建新目录。清单改变时必须更新对应SHA256及环境镜像映射；`configs/reset_audit.json`是已有题目历史reset验收证据，不能作为新增题目的已验收证明。运行目录中的配置会记录指纹，不支持静默改写后继续执行。

每轮主要产物：

| 文件/目录 | 内容 |
|---|---|
| `config.json`、`effective_config.json` | 本轮完整生效配置 |
| `environment_config.json` | 本轮工具输出和判题产物路径 |
| `launch_provenance.json`、`launch_command.json` | 配置来源、指纹、初始化权重和命令 |
| `service_checks.json` | 实际服务及文件检查 |
| `launcher.log`、`environment.log` | 主进程和环境服务日志 |
| `model_audit.json` | 新初始化adapter的短输入审计 |
| `loaded_sft_model_audit.json` | 加载真实SFT权重后的短输入审计 |
| `long_input_precheck.json` | 32K+1反向用时、梯度和峰值PyTorch显存 |
| `rollout_smoke.json` | R16候选、E/U短生成结果 |
| `summary.json`、`traceback.log` | 阶段/最终结果和失败堆栈 |
| `generations.jsonl`、`memory_validation.jsonl` | E/U原始生成和解析记录 |
| `api_requests.jsonl`、`api_responses.jsonl` | Solver/Judge请求及返回 |
| `solver_parallelism.json`、`solver_dispatch.jsonl` | Solver端点、worker编号、执行状态与并行边界 |
| `rl/solver_trajectories/` | mini-swe-agent原生轨迹、版本、实际补丁及stderr日志 |
| `e_preprocessing/` | 轨迹压缩报告与E输入 |
| `tool_outputs/`、`official_evaluator/` | 工具完整输出与官方判题证据 |
| `rl/`、`checkpoints/`、`frozen_test.json` | rollout、更新、权重和冻结测试结果 |

统一入口用文件锁避免同一入口同时训练GPU1；绕过入口的其他进程仍可能占用GPU，所以启动前查看`nvidia-smi`。不自动停止别人的进程。中断本命令会清理自己启动的环境服务；被硬杀时可能需要人工检查残留服务和容器。

## 4. 当前训练配置

默认不是30+12完整实验，也不是此前的3+2+1。包内清单是现有**5个主任务＋2个probe任务＋1个冻结测试任务**，一个计划、一个batch。两个全局replica、局部候选和比较会重复运行Solver，因此不等于只调用Solver八次。

| 部分 | 题目ID |
|---|---|
| main，按顺序 | `django__django-11740`、`15104`、`15561`、`15252`、`15268`（后四项同为`django__django-`前缀） |
| probe | `django__django-15278`、`django__django-14765` |
| test | `django__django-14122` |

| 配置 | 当前值 |
|---|---|
| 基座 | `/home/ubuntu/coding/models/Qwen3.5-4B`；revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` |
| RL初始化权重 | `/home/ubuntu/coding/MemRL/03_实验结果/checkpoints/E文本SFT30_RL初始化权重` |
| LoRA | E/U/R独立adapter，rank=8；完整248个目标层在配置中，实际兼容性由checkpoint加载校验 |
| train_roles | E、U、R；Solver/Judge/Embedding冻结 |
| dtype、device | BF16、`cuda:1` |
| E协议 | `e-text-v1`，0–3条文本记忆或单独`NO_MEMORY`；U仍为严格事务JSON |
| E轨迹输入 | `swe-evidence-hybrid-v1`预处理，上限32768；不是每条都填到32K |
| U输入、E/U输出 | U8192输入；E2048输出、U2048输出 |
| E/U采样温度 | 1.0 |
| R | HF候选token读出，合法集合重新归一化、无放回选择；默认12候选、最多选2条、预算1024token |
| 显存措施 | gradient checkpointing、CPU activation offload、MLP1024分块、逐token512分块、HF全词表归一化分块 |
| 注意力 | PyTorch SDPA `flash-gqa`，不是必须安装独立`flash-attn`包 |
| KV cache | **开启**（默认方法三），原因及近似见第5节 |
| 学习率、microbatch | E/U/R均1e-5；microbatch=1；max_grad_norm=1.0 |
| 损失相关 | policy_loss=single_step_pg；divisor=768；behavior_tolerance=0.03仅诊断；epsilon=0.2仅供显式PPO模式使用 |
| 全局与局部 | global.replicas=2；anchors_per_replica=1；I=2、J=2、K=2、L=2、B=1 |
| 主轨迹并行 | global.main_parallel_workers=2，固定每个端点一条lane；HF串行，整组结束后更新 |
| Solver主任务预算 | 最多250次模型调用、单次响应32768token、wall1200秒、命令60秒 |
| Solver局部分支预算 | 最多8次模型调用、单次响应32768token、wall180秒、命令60秒 |

局部Judge比较之前仍需运行短Solver分支，达到8次调用或180秒即截断，保留部分轨迹和补丁交给Judge，记录`probe_cutoff`；它不代表官方判题失败。本文不承诺该任务流的完成时间或正确率。

SFT转RL时仅导入E/U/R权重，重新初始化优化器、游标和记忆库。`--prepared-root`只是接着使用准备目录，不是断点续训；已有RL恢复应使用对应配置的CLI `train CONFIG --resume CHECKPOINT`，且需保持协议与路径一致，不能重新走SFT初始化覆盖它。

## 5. HF token概率检查：0.03具体是什么意思

0.03原本是严格PPO路径的训练正确性门槛。当前默认方法三仍计算差异，但超过0.03只记录，不停止；它不是当前PG损失的分母。生成时保留实际token及原始logprob，训练再把**同一段输入、同一串已选token**交给HF，重算这些token的logprob。E/U在有效位置诊断最大绝对差：

```text
max(abs(logprob_生成时 - logprob_更新前重算)) <= 0.03
```

它不是再生成另一篇文章后比较文字，不是“允许3%的token不同”，也不是正确率阈值。大模型抽样有随机性，但一旦选中的token已经固定，这里检查的是同一动作在同一策略下的概率是否能重现。

logprob是自然对数概率。单token差0.03对应概率比最多约`exp(0.03)=1.03045`。这是单位置的检查，**不能推出长文本整段联合概率只差3%**。R在训练中另比较所选slate的联合logprob；R全词表读出与候选投影还有独立检查，不要把它们混为同一个阈值。

为什么纯HF仍可能失败：Qwen3.5包含线性注意力/递归状态；逐token缓存生成和整段训练重算可能走不同计算路径，BF16及内核也可能产生数值偏差。“都是HF”本身不保证两个路径一致。这与早先HF/vLLM跨后端差异属于不同问题。

已知实测：

| 设置 | 记录 |
|---|---|
| HF生成开启KV cache | 短输入审计最大logprob差约0.1234536，大于0.03，启动被拒绝 |
| HF生成关闭KV cache | 此前短输入E约2.16e-7、U约1.91e-6，检查通过 |

当前默认配置为`generation.use_generation_cache=true`、`optimization.policy_loss=single_step_pg`。E/U使用`-advantage * sum(有效回答token的重算logprob) * row_weight / divisor`；R使用同样权重乘所选slate的联合logprob。没有新旧概率比、PPO裁剪或混合后端修正；不覆盖原始old_logprobs。每批完整采样后积累所有梯度，再各优化器step一次。缓存生成与整段重算仍可能不同，因此属于接受该差异的近似，不能保证无偏或学习有效。

2026-10-15方法三短输入实测：加载当前SFT权重后，启动审计通过，E（366输入＋4输出）最大logprob差约0.00006215，U（130输入＋4输出）约0.011827，均未超过0.03。审计耗时约103秒，未运行做题或optimizer step。此前开启cache超限的结果属于不同测试条件，不能用本次短输入结果推断长输入也对齐。记录在`03_实验结果/diagnostics/method3-default-20261015`。

`rl/probability_diagnostics.jsonl`持久记录E/U/R概率差异。E/U记录row_id、输入/输出/总长度、重算耗时、差异均值/最大值/p50/p95/p99及超阈值比例；backward事件按row_ids关联反向耗时和本次update以来峰值显存（不是该条样本独占显存）。microbatch>1时耗时属于整组，不当作单样本耗时。R差异单位是joint slate，不是单token。更新结果也保存同一诊断列表。

启动审计报告区分`passed`和`behavior_alignment_passed`：方法三可在有限误差超过0.03时通过启动，但后者为false。非有限概率/梯度、错误版本、基础权重变化、梯度隔离及R读出正确性仍会报错。CPU offload和分块配置继续保留，缓存不解决训练反向OOM。

需要旧严格方案时，在新完整配置中显式设置`optimization.policy_loss=ppo`；统一入口将关闭缓存并恢复概率差异超限停止。不要修改既有运行目录来切换协议。未填写policy_loss的新配置默认补single_step_pg；历史RL协议hash变化后不能直接续训，需明确选择并验证旧协议或开始新的运行。

启动阶段检查新adapter，也检查加载SFT后的真实adapter；训练阶段继续检查实际采样记录。短输入审计通过，不保证所有长输入都通过。32K容量预检只用**32768输入＋1个目标token**；它不证明32K＋2048完整输出一定不OOM，也不验证长输出的所有token概率。

显式PPO模式下，出现`Generation behavior drift`或`E/U behavior/recompute logprob mismatch; no update published`仍停止；默认PG模式对有限差异只记录。任何真正异常都会保留日志并回滚本次adapter及优化器，不发布该次更新checkpoint。先核对本轮policy_loss，不把诊断阈值超限和数值NaN、OOM等故障混为一谈。

## 6. 服务启动、复用与恢复

先检查已运行的服务，健康时直接复用。下列脚本均以前台方式运行，便于放入tmux并保留日志；启动前确认相应GPU和端口未被其他服务占用。不会在安装或导入源码时隐式启动模型。

### coding：Solver / Judge

```bash
cd /home/ubuntu/coding/MemRL/02_源码与脚本
bash scripts/services/solver.sh /home/ubuntu/coding/models/Qwen3.5-4B 8000
```

固定使用`/home/ubuntu/coding/venvs/vllm/bin/vllm`，GPU0、BF16、max_model_len196608、显存预留0.80、max_num_seqs1、Qwen3 reasoning parser及qwen3_coder工具解析。具体参数见脚本。不要在原8000服务健康时再启动一份。

### 远端：Embedding

将源码包或`embedding.sh`放在远端可访问位置，在远端终端运行：

```bash
bash /实际源码路径/scripts/services/embedding.sh \
  /defaultShare/archive/chenzhenyuan/code/LLM/OpenSource/Qwen/Qwen3-Embedding-8B 18082
```

固定环境`/defaultShare/archive/chenzhenyuan/anaconda3/envs/llm/bin/python`，GPU1、BF16、pooling LAST、L2激活、max_model_len32768、显存预留0.90。脚本缓存目录沿用`/defaultShare/archive/chenzhenyuan/code/cyx/memrl-mixed-20261013`下的cache/tmp；运行账号需能读写。

coding的8002未连通但远端18082健康时，恢复已有转发。包中`scripts/services/embedding_tunnel.sh`仅转发Embedding，不再转发旧E/U/R端口。它从远端执行，参数为现有可用coding SSH目标和必要SSH选项；认证与登录由接收人现有配置提供，包不包含凭据。不要同时建立争用8002的两条隧道。

### coding：SWE环境与官方评测

通常使用`--start-environment`即可，脚本自动以本轮配置启动18119并检查配置指纹。若已有18119服务但配置不匹配，明确报错，不自动终止别人的服务。

单独开环境服务时：

```bash
bash scripts/memrl.sh --prepared-root /本轮已准备目录 --serve-environment

# 另一个终端，使用同一目录：
bash scripts/memrl.sh --prepared-root /本轮已准备目录 --train
```

等价底层命令如下，配置是位置参数：

```bash
export PYTHONPATH=/home/ubuntu/coding/MemRL/02_源码与脚本/src
/home/ubuntu/coding/venvs/eur-alf-pilot-20261011/bin/python \
  -m eur_memory_rl serve-environments /本轮目录/environment_config.json \
  --host 127.0.0.1 --port 18119
```

`GET /health`只返回服务标识、配置指纹和grader revision，不创建容器。`--environment-smoke`另验证实际Docker任务链路。正式grader使用`/home/ubuntu/coding/venvs/swebench/bin/python`、900秒测试预算、1800秒外层预算；数据源为`/home/ubuntu/coding/MemRL/03_实验结果/datasets/evaluator_protocols/verified-legacy-to-v5-20261007-01/verified.evaluator.jsonl`。工具容器无网络、4CPU、8g内存，工具输出展示限制20000字符，完整输出归档。本地8个digest锁定镜像在`configs/environment.json`中。

## 7. 模型调用示例

以下命令在 **coding 的 Linux 终端**执行。Solver 和 Judge 共用 `qwen3.5-4b`；Embedding 通过 coding 的8002端口调用远端模型。当前无需API密钥。

### Solver

```bash
curl --noproxy '*' http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.5-4b",
    "messages": [{"role": "user", "content": "请简短解释二分查找。"}],
    "max_tokens": 256,
    "temperature": 0.6,
    "chat_template_kwargs": {"enable_thinking": false}
  }'
```

正文位于 `choices[0].message.content`。此例关闭思考以快速验证调用；正式Solver配置开启思考，输出预算为32768。

### Judge（同一个模型服务）

```bash
curl --noproxy '*' http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.5-4b",
    "messages": [{"role": "user", "content": "只返回JSON对象：{\"result\":\"OK\"}"}],
    "max_tokens": 128,
    "temperature": 0,
    "response_format": {"type": "json_object"},
    "chat_template_kwargs": {"enable_thinking": false}
  }'
```

### Embedding

```bash
curl --noproxy '*' http://127.0.0.1:8002/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3-embedding-8b",
    "input": ["SQLite迁移时新增唯一字段失败的排查经验。"]
  }'
```

向量位于 `data[0].embedding`，维度为4096。

### E 的标准输出格式（当前SFT与RL共用）

当前E使用`e-text-v1`协议。SFT用于学习按这个格式输出；加载SFT权重后继续使用相同提示词与解析规则，不保证模型每次都能正确遵守。

**一条记忆：**

```text
[MEMORY]
正文：修改字段属性时，先判断它是否会改变数据库结构，再检查对应的迁移操作。
使用条件：适用于已观察到的Django字段迁移问题；具体行为仍需结合数据库后端验证。
```

**多条记忆：**一次调用E最多输出3条，每条重复同一结构，不需要调用多次E。

```text
[MEMORY]
正文：排查SQLite迁移失败时，应检查新增字段的唯一性约束及生成的SQL。
使用条件：适用于SQLite新增唯一字段时出现的迁移错误，不直接推广到其他后端。

[MEMORY]
正文：修改迁移生成逻辑后，应运行针对该场景的回归测试，并检查实际生成的数据库操作。
使用条件：适用于迁移生成逻辑的修改；仅有补丁或模型自述不能证明测试通过。
```

**没有值得写入的记忆：**只输出下面这一行。

```text
NO_MEMORY
```

以上是格式示例，不是本次模型实测生成内容。E只输出正文和使用条件，不输出JSON、kind、source_ids或额外解释，也不需要在输出外包Markdown代码围栏。不要在有效记忆后追加`NO_MEMORY`；这属于矛盾输出，代码会拒绝。

程序把文本包装成内部候选，补充来源信息；kind当前使用内部`unclassified`。这些是E的候选记忆，是否真正写入以及如何更新已有记忆，仍由后续U处理。**E的格式变化不改变U的事务JSON协议。**

## 8. 环境、资源、数据与扩展操作

| 用途 | 已有环境 |
|---|---|
| HF采样/训练/SFT | `/home/ubuntu/coding/venvs/eur-alf-pilot-20261011`；Python3.10.12、torch2.10.0、Transformers5.19.0、PEFT0.21.2、causal-conv1d1.7.0 |
| coding Solver | `/home/ubuntu/coding/venvs/vllm`；Python3.10.12、vLLM0.17.1 |
| 官方SWE评测、mini-swe-agent | `/home/ubuntu/coding/venvs/swebench`；mini-swe-agent2.4.6 |
| SWE-agent独立工具 | `/home/ubuntu/coding/venvs/sweagent`；Python3.11.16、SWE-agent1.1.0；不是mini-swe-agent环境 |
| 远端Embedding | Conda `/defaultShare/archive/chenzhenyuan/anaconda3/envs/llm`；Python3.10.20、torch2.10.0、Transformers5.8.0、vLLM0.19.1 |

当前coding为双RTX3090，各24GiB；WSL内核6.18.54.1，WSL可见内存约62.7GiB、swap16GiB。CPU offload需要这部分内存与CPU/GPU传输。32K通过不代表任意输入输出组合均可训练；OOM时保留日志，先检查其他进程和实际输入/输出长度，再决定减少预算或优化证据输入，不能把静默截断当作完整训练。

包内`configs/dependency_lock.json`为现有验收依赖锁，启动preflight会核对安装版本。其platform字段来自早期导出，不能用它判断当前WSL内核。`pyproject.toml`声明Python>=3.11，而现有验收环境为3.10；当前通过已有Python和源码路径使用，不执行`pip install -e .`或升级现有依赖。`eur-fast-ops-20261014`不是完整venv。

权限要求：能读取模型和checkpoint、写入独立运行目录、使用GPU及Docker、执行官方评测Python、访问模型API。更换用户后应实际检查这些权限，不能仅凭目录存在判断可用。

| 既有数据 | coding路径 |
|---|---|
| no-memory500 | `/home/ubuntu/coding/MemRL/03_实验结果/datasets/baselines/nomemory500` |
| ReasoningBank220 | `/home/ubuntu/coding/MemRL/03_实验结果/datasets/baselines/reasoningbank220` |
| ReasoningBank30+12 | `/home/ubuntu/coding/MemRL/03_实验结果/datasets/baselines/reasoningbank30plus12_repeat` |
| 文本E SFT教师集 | `/home/ubuntu/coding/MemRL/03_实验结果/datasets/sft/memrl-e-text-sft30-20261015-r2-data/teacher_bundle/teacher`；24train、6dev及semantic_review.json |

独立压缩原始轨迹（只加载tokenizer，不生成/训练模型）：

```bash
PY=/home/ubuntu/coding/venvs/eur-alf-pilot-20261011/bin/python
"$PY" scripts/compact_swe_hybrid.py --trajectory /实际轨迹.traj.json \
  --tokenizer /home/ubuntu/coding/models/Qwen3.5-4B \
  --prompt-profile e-text-v1 --max-input-tokens 32768 --output /新的压缩输出目录
```

SFT复现实验使用包内`run_text_e_sft.py --help`查看参数：`--config`完整配置、`--teacher`上述教师目录、`--checkpoint`受信任的原始初始化checkpoint、`--root`新结果目录。当前交付默认直接复用已完成的SFT权重，无需重做SFT；不要误把已SFT权重当作原始起点后宣称重现同一实验。工具CLI、SFT、预处理入口均在源码中，默认训练只由统一入口执行。包中不包含Codex认证或教师生成密钥。

## 9. 验收记录与边界

2026-10-15已完成以下验收。源码测试、服务检查、环境工具检查和真实HF启动分别记录；不把启动验收等同于完整RL实验，也不承诺记忆有效性或做题正确率。

| 验收 | 结果 |
|---|---|
| 本地核心源码和启动入口测试 | 153项通过；修改入口的静态检查、shell语法检查通过 |
| 源码包解压后，coding现有Python运行完整tests | 153项通过；`--help`和新目录准备成功 |
| 包内配置和依赖 | 任务清单hash、基座、真实SFT权重、grader数据/解释器路径存在；8个题目镜像本地可用 |
| 模型调用 | Solver实际返回`OK`；Embedding实际返回4096维有限向量，范数约1.0000000008 |
| 环境服务 | 自动启动与清理成功；真实题目reset、只读命令、finalize、close通过，初始hash匹配、补丁为空 |
| 配置隔离保护 | 使用另一运行目录连接现有环境服务时被明确拒绝；使用同一准备目录检查通过 |
| 新adapter模型审计与完整Runtime | 通过 |
| 真实SFT权重加载后审计 | E最大logprob差8.94e-8，U为5.36e-7，R读出差0；梯度隔离通过 |
| 长输入反向 | 32768输入＋1目标token，约184.18秒，峰值allocated14,901,854,208字节，约13.88GiB；E梯度范数85.35，U/R为0 |
| HF采样检查 | R16候选、E/U各最多8token通过；不代表完整记忆格式/质量验收 |
| 整轮启动 | `startup_passed`，约390.11秒；task_rollouts=0、optimizer_steps=0、completed_updates=0 |

本轮证据目录：`/home/ubuntu/coding/MemRL/03_实验结果/runs/memrl/handoff-validation-20261015`。其中`environment_smoke.json`、`extracted_archive_checks.json`、`loaded_sft_model_audit.json`、`long_input_precheck.json`、`summary.json`及日志可交叉核对。

测试结束后，本轮自动启动的18119服务已停止，避免占用后续运行的端口；接收人使用`--start-environment`启动自己的本轮服务。Solver未重启，GPU1训练模型已退出并释放资源。远端Embedding通过coding转发API验收，没有重启远端服务或改变登录配置。正式5+2+1训练、官方判题完整执行、SFT重训和长输出概率一致性未在本次交付验收中执行。

### 目录整理后的复验

唯一源码位于`02_源码与脚本`，部署配置位于`01_部署与配置`，数据、权重、运行和诊断位于`03_实验结果`。开发及清理约定见同目录`开发与实验规范.md`；实验索引见`../03_实验结果/实验索引.md`。不再维护独立共享源码目录或`.bak`副本，旧源码通过Git的`history/*`标签追溯。

迁移后153项测试通过，模型API与真实工具环境检查通过；真实HF启动验收耗时386.90秒，32K输入反向耗时181.82秒，峰值allocated约13.88GiB。没有正式题目rollout或optimizer更新。证据位于`03_实验结果/diagnostics/20261015-layout-startup`。500条no-memory轨迹路径全部有效，内部软链接无失效项；整理清单位于`diagnostics/file_organization_20261015`。32K容量检查仍仅包含1个目标token，不代表长输出训练已验收。


## 夜间五组训练协议

30道训练题预声明为五组，每组4道主任务、2道局部probe；每组2个replica，E/U/R权重与优化器连续更新5次。历史协议各组从空记忆库开始并在最后重建测试记忆库；2026-10-16第一组完成后，后四组改用下述连续记忆协议，不改写历史记录。

统一入口增加 `--train --start-environment --full-suite`；完整命令须使用本轮准备完成的独立配置和输出目录，不复用历史运行目录。`--full-suite`不是RL断点续训开关。

Judge可显式启用 `structured_output=true` 和 `repair_attempts=2`，非零评分仍要求工具证据。`local.pin_probe_backend=true`将同一probe的参考和所有候选固定到同一Solver；不同probe可并行。`local.min_legal_candidates=2`只根据采样前公开记忆库筛选节点，不保证不同动作或非零奖励；没有合格节点时停止排查。`schedule.require_declared_plans=true`禁止失败后自动替换预声明主组。


### 分组环境准备与Windows下载

`prepare_grouped_swe.py --deferred-environments`预先冻结全部题目和镜像内容摘要；`environment.readiness_mode=per-group-v1`要求每组开始前对该组主任务及probe各做两次独立干净重置。未缓存的后续镜像可以同时下载；该组开始时仍未齐则最多等待两小时，不能跳题。测试环境在统一测试前同样验收。`reset_audit.json`明确记录分组模式，不把尚未执行的检查标成通过；实际证据写入运行目录`rl/environment_readiness.json`。

`scripts/windows_registry_bridge.py`提供仅监听WSL回环地址的临时镜像转发、Windows原生curl下载worker和Docker现有containerd命名空间导入。Windows文件缓存位于`%TEMP%\MemRL-registry-cache`；下载后Windows与WSL分别验证SHA256。此方式不修改Docker守护进程或全局代理配置。仅缺失镜像需要新下载，已有镜像层继续复用。

完整RL断点恢复使用`--train --start-environment --full-suite --prepared-root 运行目录 --resume-checkpoint 运行目录/checkpoints/完整checkpoint`，恢复权重、优化器、RNG及数据游标；只接受同一源码和协议。SFT初始化仍是新优化器、新游标。


### WSL CPU卸载的进程资源限制

SSH默认打开文件数软上限1024可能在32K反向CPU activation offload时耗尽，内核表现为`dxg get_unused_fd_flags failed: ffffffe8`并伴随CUDA unknown及EMFILE。统一启动脚本仅将自身进程树软上限提高到65536（受硬上限约束），不改系统配置。显存、内存与文件句柄上限是不同资源，不能把此错误直接记为OOM。

### U 专项 SFT（2026-10-16）

通过同一入口运行 `bash scripts/memrl.sh --u-sft --config 配置.json --teacher 已审阅教师目录 --checkpoint 已接受的RL前初始化目录 --root 新结果目录`。默认仅准备并校验数据；显式加 `--train` 才加载 coding GPU1 并训练。需要原教师目录的 train.json、dev.json、semantic_review.json，以及配置旁的 public_manifest.json 用于来源隔离核验。

本入口始终将 U 恢复成原始 Qwen3.5-4B 底座上的新 LoRA（初始 B 矩阵为零），仅从 checkpoint 导入 E/R；不继承 checkpoint 的 U 或任何优化器。新建 U AdamW，默认 lr=1e-4，仅一轮。旧示范加显式标记的独立记忆保留合成场景；训练前后验证固定 dev，核对 U 改变而 E/R/底座不变。使用与 RL 相同 GPU 文件锁，不会停止其他服务，不自动发布为默认权重或启动 RL。

当前运行目录为 `03_实验结果/runs/sft/u-sft-20261016-r1`。设计与数据边界见 `01_部署与配置/historical_reports/2026-10-16_U专项SFT.md`；实际完成状态以该运行目录的 summary.json 为准。

### 连续记忆、分组更新（2026-10-16）

`memory_continuity.enabled=true` 时，每个 replica 继承自身上一组的主轨迹记忆库；两条流不合并、不交换记忆。局部候选树是反事实评估，不写回主库。每组所有主轨迹和局部树完成后，各角色仅更新一次，将两份完整记忆库、优化器、RNG和训练游标共同保存到 checkpoint。重启时恢复，不依赖进程内缓存。

本轮从已完成的第一组切换协议，第二组旧尝试停止且未更新。迁移是显式新运行，保留第一组 E/U/R、优化器、RNG和 cursor=1，并从原始 rollout 读取各 replica 的主库。迁移工具不调用模型：

```bash
cd /home/ubuntu/coding/MemRL/02_源码与脚本
/home/ubuntu/coding/venvs/eur-alf-pilot-20261011/bin/python scripts/fork_continuous_memory.py \
  --source-root /home/ubuntu/coding/MemRL/03_实验结果/runs/memrl/20261016-mini-dual-sft-five-groups-r2 \
  --iteration 1 \
  --root /home/ubuntu/coding/MemRL/03_实验结果/runs/memrl/20261016-mini-dual-continuous-memory-r1
```

工具输出新的完整 checkpoint 路径；用统一入口 `bash scripts/memrl.sh --prepared-root 新运行目录 --train --full-suite --max-updates 2 --resume-checkpoint 新checkpoint路径 --start-environment` 启动第二组。完成后核验新 checkpoint，再分别将累计上限改为3、4、5，每次恢复上一组完整 checkpoint。

新协议后四组的首题必须带已有记忆真实调用 Solver，不再复用 no-memory 轨迹。最后12道测试固定使用最终 replica-0 的主库，不重建、不合并、不按测试结果选库；测试期间库和参数均冻结。奖励只归因于当前组，旧组决策不会收到后续组的追溯梯度。记忆增多扩大可选状态，不保证正确率或非零优势必然提高。

源/目标协议 hash、checkpoint SHA256、两份库的 hash 和条目数写入 `protocol_transition.json`；每组起止库写入 `rl/memory-continuity-XXXXXX.json`，checkpoint 的 `data_state.memory_banks` 保存完整状态。旧运行配置、rollout 和 checkpoint 不被覆盖。
