# MemRL

这是coding上的唯一开发源码。项目总目录为`/home/ubuntu/coding/MemRL`。

- [部署与运行说明](使用说明.md)：模型API示例、E标准输出、训练参数及HF的0.03概率检查。
- [开发与实验规范](../01_部署与配置/开发与实验规范.md)。
- 配置：`configs/`指向`../01_部署与配置/experiments/`。
- 数据/权重/运行/诊断：`../03_实验结果/`。
- 当前交付：`releases/current/`，由打包脚本生成，不维护共享源码副本。

在coding终端：

```bash
cd /home/ubuntu/coding/MemRL/02_源码与脚本
bash scripts/memrl.sh --help
bash scripts/memrl.sh --check

# 默认仅验收，不执行做题/参数更新。
bash scripts/memrl.sh --startup-only --start-environment

# 明确开始下一轮完整实验时再运行。
bash scripts/memrl.sh --train --start-environment
```

环境API未运行时，单独`--check`会报告失败；可按使用说明先准备目录，再使用`--check --start-environment`临时启动匹配的环境。

脚本用途：`run_hf_orm.py`为内部HF任务流，`run_text_e_sft.py`复现文本E SFT，`compact_swe_hybrid.py`及`verify_swe_hybrid_outputs.py`处理/验证轨迹，`call_model.py`直接调用外部模型，`resolve_artifact.py`解析迁移前的路径，`services/`包含正式服务入口。旧混合后端入口及一次性队列/修补脚本已从工作目录移除，历史源码见Git标签。
