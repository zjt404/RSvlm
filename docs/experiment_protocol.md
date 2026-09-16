# 实验协议与检查点

## 对比对象

| 展示名称 | 实际检查点 |
|---|---|
| Baseline | 基础 Qwen3-VL-4B，无本项目适配器 |
| SFT | `outputs/sft_context_v6_dense/v0-20260909-163120/checkpoint-512` |
| GRPO | `outputs/grpo_v8_mixed/v0-20260914-171138/checkpoint-625` |

最终 SFT 由通用监督微调、关系全框专项训练、密集目标补充训练顺序得到。表中仅展示最终 SFT，不表示历史上只进行过一次训练。

## 固定测试协议

- 通用测试：`data/processed_fixed/sft_test.jsonl`，4,000 条。
- 全框测试：`data/processed_context_v6_dense_sft/sft_context_test.jsonl`，2,666 条。
- 推理使用 `rs_vlm.predict`、`do_sample=False`；通用任务最大新 token 数 512，全框任务 1024；均不额外设置 max_pixels。
- 全框评分使用 `scripts/evaluate_context_all_boxes.py`，按 IoU 降序贪心一对一匹配、阈值 0.5；计数由有效框数得到。
- 通用评分使用 `rs-vlm-eval`，不同任务分别报告。

全框 JSON 有效率采用容错解析，允许从文本中抽取 JSON，并不等价于严格裸 JSON 率。Schema 检查键集合、坐标与计数一致性。现有评分器未单独输出分桶 JSON/Schema 率。

## 结果来源

以下为训练服务器项目目录下的原始指标文件，未随代码发布：

- Baseline 通用：`outputs/evaluation/base_metrics.json`
- SFT 通用：`outputs/evaluation_general_retention_v1/v6_general_metrics.json`
- GRPO 通用：`outputs/evaluation_general_retention_v1/grpo_v8_general_metrics.json`
- SFT 全框：`outputs/evaluation_context_v6_dense/v6_relation_metrics.json`
- GRPO 全框：`outputs/evaluation_context_v6_dense/grpo_v8_relation_metrics.json`

Baseline 尚无最终全框协议下的完整结果，因此保留缺失值。

## 解释边界

结果为单次运行点估计，未完成多种子、显著性或非劣效检验。开发过程中曾参考测试结果调整方案，该测试集并非完全未接触的最终盲测集；泛化结论还需独立留出数据验证。训练奖励对不确定标注采用忽略处理，正式全框评分仍按固定目标框统计误检。
