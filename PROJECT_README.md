# 运行指南

## 环境

安装适合 GPU 的 PyTorch/CUDA 后：

```bash
pip install -e '.[train,app,dev]'
python -m pytest -q
```

现有服务器环境使用 `source scripts/activate_qwenvl.sh`。脚本包含实验服务器路径，迁移时须调整。模型、数据和适配器需要单独准备。

## 数据与训练

通用数据转换和校验使用 `rs-vlm-data --help`。下载及准备入口为 `scripts/download_vrsbench.sh`、`scripts/download_dior.sh`、`scripts/prepare_data.sh`。

最终权重的实际训练顺序：

1. `scripts/train_sft.sh`：通用多任务 SFT。
2. `scripts/prepare_context_v5_all_boxes.py`：关系全框数据及通用回放；专项训练入口为 `scripts/train_sft_context_stage2.sh`，须设置 TRAIN_DATA、VAL_DATA、BASE_ADAPTER、OUTPUT_DIR。历史默认值不代表最终专项运行配置，精确复现需使用原始 args.json。
3. `scripts/prepare_context_v6_dense_sft.py` 与 `scripts/train_sft_context_v6_dense.sh`：加入密集样本并继续 SFT，得到结果表中的 SFT。
4. `scripts/prepare_grpo_v8_mixed.py` 与 `scripts/train_grpo_v8_mixed.sh`：混合密度及通用回放 GRPO。

数据构建参数使用脚本的 `--help` 查看。最终 GRPO：2,500 条样本，1 epoch，学习率 1e-6，KL 系数 0.04，每提示 4 个候选，LoRA rank 16 / alpha 32。完整原始运行配置和权重未随仓库发布。

## 单图测试

```bash
python scripts/chat_v8.py --model /path/to/model --adapter /path/to/adapter --image /path/to/image.jpg --question "图中是否有船只？" --task vqa
```

使用 `--boxes` 请求全框输出；`--task count`、`grounding`、`spatial` 分别测试其他任务。不传任务参数时直接使用问题。每次调用重新加载模型，该入口是单图单轮测试。

## 统一评测

```bash
# Baseline 不传 --adapter；SFT、GRPO 使用对应适配器。
python -m rs_vlm.predict --dataset data/processed_fixed/sft_test.jsonl --model /path/to/model --adapter /path/to/adapter --max-new-tokens 512 --output outputs/general.jsonl
rs-vlm-eval outputs/general.jsonl --output outputs/general_metrics.json

python -m rs_vlm.predict --dataset data/processed_context_v6_dense_sft/sft_context_test.jsonl --model /path/to/model --adapter /path/to/adapter --max-new-tokens 1024 --output outputs/relation.jsonl
python scripts/evaluate_context_all_boxes.py outputs/relation.jsonl --output outputs/relation_metrics.json
```

为各模型指定独立输出文件；预测入口会覆盖同名文件，不支持断点续评。比较时固定提示、图像和解码设置。

## 服务与审计

服务入口为 `rs-vlm-api` 与 `rs-vlm-demo`，使用 `--help` 查看参数。误检分析和可视化使用 `scripts/analyze_all_boxes_false_positives.py` 与 `scripts/render_fp_proximity_audit.py`。audit 工具用于标注检查。`scripts/evaluate_context.py` 保留为旧标注协议解析测试的依赖，最终全框指标使用 `evaluate_context_all_boxes.py`。
