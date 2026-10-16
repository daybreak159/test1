# MemRL

本目录是 coding 开发源码的筛选同步版，来源提交为
`8ecf5b4813b5b51dd78c91510ab6a6a986bb6bee`。保留选中文件的 14 个开发提交；
新旧提交对应关系见 [SOURCE_HISTORY.md](SOURCE_HISTORY.md)。

## 实现范围

- E 提取正文与使用条件，U 管理记忆事务，R 在合法集合中选择记忆。
- 原始工具轨迹预处理和有损压缩，限制 E 的输入长度。
- Solver 通过独立 worker 接入真实 mini-swe-agent，使用 ReasoningBank 提示词。
- 两个独立 Solver replica 并行执行，HF E/U/R 调用串行，共同完成后更新参数。
- 主任务结果奖励、短局部分支与 Judge 比较、single_step_pg 更新。
- E/U SFT 初始化、逐组 checkpoint 恢复，以及独立 replica 记忆库跨组继承。

## 源码结构

| 位置 | 职责 |
|---|---|
| `src/eur_memory_rl/data/` | 轨迹处理、数据协议和历史首题复用 |
| `src/eur_memory_rl/domains/` | Solver、mini-swe-agent 适配及提示词 |
| `src/eur_memory_rl/memory/`、`policies/` | 记忆事务、E/U 生成和 R 选择 |
| `src/eur_memory_rl/rollout/`、`rewards/` | 主流程、局部比较与奖励 |
| `src/eur_memory_rl/training/` | HF 反向、概率诊断、优化器及 checkpoint |
| `src/eur_memory_rl/services/` | 执行环境服务与隔离 |
| `start_memrl.py`、`scripts/memrl.sh` | 验收、训练与恢复入口 |

`scripts/` 仅保留 8 个入口：`memrl.sh`、`run_hf_orm.py`、`run_text_e_sft.py`、
`run_u_sft.py`、`compact_swe_hybrid.py`、`verify_swe_hybrid_outputs.py`、
`prepare_grouped_swe.py`、`fork_continuous_memory.py`。

## 运行前提

此仓库只同步源码，不包含测试目录、基线源码目录、实验数据、模型权重、配置、
服务部署脚本或实验日志；这些内容也已从导入的历史中排除。
核心源码保持 coding 版本，不因整理而修改算法或正在运行的实验。

当前启动脚本使用 coding 的既有 Python 环境和目录，不能把新拉取的仓库直接当作
已配置的运行环境。执行需要另行准备实际配置、数据清单、模型、SFT/RL checkpoint、
Docker 任务环境和模型服务。mini-swe-agent 安装于独立 Python 环境，当前部署版本为
`2.4.6`；HF 与服务等依赖见 `pyproject.toml`，实际版本锁由外部运行配置提供。

在已经配置好的 coding 环境，可查看统一入口：

```bash
cd /home/ubuntu/coding/MemRL/02_源码与脚本
bash scripts/memrl.sh --help
```

默认只做启动验收。正式训练必须显式使用 `--train`；SFT 权重导入与完整 RL
断点恢复是不同操作。当前 E/U/R 为纯 HF，KV cache 开启，默认每批各优化器仅
step 一次。有限概率差异的 `0.03` 阈值只作诊断，不阻止默认 PG 更新。

开发约定见 [AGENTS.md](AGENTS.md)。开发临时输出放 `_dev/`，只跟踪其 README。
