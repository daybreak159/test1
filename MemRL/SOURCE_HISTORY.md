# 源码历史来源

来源：coding `/home/ubuntu/coding/MemRL/02_源码与脚本` 当前分支，
截至 `8ecf5b4813b5b51dd78c91510ab6a6a986bb6bee`。

在独立克隆中按白名单保留 `src/`、8 个正式脚本、`start_memrl.py`、
`pyproject.toml`、`README.md`、`AGENTS.md`、`.gitignore`，并将历史路径加上
`MemRL/` 前缀。只导入当前分支的祖先提交，不导入独立历史快照标签。
coding 原始仓库、文件和提交历史保持不变。

导入后另作同步范围提交，加入最新忽略规则、本文和源码版说明；
不修改核心源码。路径和内容筛选会改变提交 SHA，作者及提交说明保留。

| coding 原提交 | 导入后提交 | 主要修改 |
|---|---|---|
| `9ec9614` | `aa44199` | 目录整理和 HF 入口 |
| `c95268d` | `798493d` | HF 缓存与单步 PG；旧交付 zip 对应此版本 |
| `f92f1bb` | `99ab9a6` | 成对奖励、分组训练与环境重置 |
| `18ed5fe` | `2dbdb81` | Solver 端点启动检查 |
| `c7a1ce1` | `94b63d0` | 完整 RL checkpoint 恢复 |
| `5151add` | `4dbc611` | 分组镜像和重置检查 |
| `e3d29d7` | `18518bf` | 按组检查镜像缓存 |
| `bbd4a3d` | `a361bf6` | WSL activation offload 文件句柄上限 |
| `e4d36e6` | `ae3af91` | U 专项 SFT |
| `5dc8d51` | `5755507` | mini-swe-agent 接入 |
| `3292b29` | `5cdf52a` | 双 Solver 并行与环境锁隔离 |
| `cf45b74` | `95a3c61` | 逐组训练和首题历史轨迹复用 |
| `950a533` | `d542b3b` | ReasoningBank 提示词与重启后历史轨迹加载 |
| `8ecf5b4` | `7f2337c` | replica 记忆库跨组继承 |
