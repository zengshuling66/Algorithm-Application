# RAG评测报告：混合检索＋精排 / dev / Top3

## 来源与已记录配置
```json
{
  "collection": "cockpit_manual_bge_small_zh_v1_5_tables_textflow_v1",
  "source_id": "training_manual_a",
  "content_hash": "d6509e2e55a17e9145b9af6883ef4629c380e7213944b9fcd8ee3b050ed7d2a1",
  "top_k": 3,
  "candidate_k": 10,
  "embedding_model": "BAAI/bge-small-zh-v1.5",
  "reranker_model": "BAAI/bge-reranker-base",
  "reranker_revision": "2cfc18c9415c912f9d8155881c133215df768a70",
  "requested_model": "deepseek-v4.1-flash",
  "review_gold_file": "C:\\Users\\Administrator\\Desktop\\OneDrive - 南方科技大学\\丁师兄训练营\\WEEK\\projects\\smart_cockpit_rag\\dev_retrieval_gold_v3.json",
  "review_gold_sha256": "eb2e26630920d8e54a506dc6c2535b99dfa6f1dcc4cdc80d7a95e7cc1d807f41",
  "review_file": "C:\\Users\\Administrator\\Desktop\\OneDrive - 南方科技大学\\丁师兄训练营\\WEEK\\projects\\smart_cockpit_rag\\dev_qa_top3_review_v4.json",
  "review_sha256": "bb7d052fc23193e11870e01db900125e11dcd2fb2b0fe10f4c467d8ebdf14b74",
  "selected_method": "rerank",
  "split": "dev",
  "report_cases": 30,
  "source_records": 120,
  "actual_models": [
    "deepseek-v4-1-flash"
  ],
  "prompt_versions": [
    "cockpit-rag-json-v2"
  ]
}
```

## 指标
| 策略 | 指标 | 数值 | 已评题数 | 待评题数 |
| --- | --- | --- | --- | --- |
| rerank | precision_at_k | 56.94% | 24 | 0 |
| rerank | required_block_recall | 100.00% | 24 | 未单列 |
| rerank | all_required_blocks_hit_rate | 100.00% | 24 | 未单列 |
| rerank | task_pass | 100.00% | 30 | 未单列 |
| rerank | key_fact_coverage | 100.00% | 24 | 0 |
| rerank | faithful | 100.00% | 30 | 未单列 |
| rerank | citation_pass | 100.00% | 28 | 未单列 |
| rerank | candidate_required_block_recall | 100.00% | 24 | 未单列 |
| rerank | required_block_mrr | 0.9167 | 24 | 未单列 |

指标定义：
- `precision_at_k`：相关返回块数除以K，再按可回答题平均；缺位不算相关。
- `required_block_recall`：必要块命中数除以金标必要块数，再按题平均。
- `all_required_blocks_hit_rate`：最终TopK找齐全部必要证据块的题目比例。
- `task_pass`：已复核题中的任务通过比例；不可回答题要求正确拒答。
- `key_fact_coverage`：答案正确覆盖事实数除以金标事实数，再按题平均。
- `faithful`：适用且已复核的答案中，实际陈述忠于上下文的通过比例。
- `citation_pass`：适用且已复核的答案中，引用支持相应陈述的通过比例。
- `candidate_required_block_recall`：最终截取TopK前，候选必要块召回率的题目平均值。
- `required_block_mrr`：各题首次命中必要证据块的RR均值；未命中为0。

## 回答与拒答
- 可回答题：共24题；已复核24题，通过24题，待复核0题；有效决策24次，拒答0次；工程错误0次。
- 不可回答题：共6题；已复核6题，通过6题，待复核0题；有效决策6次，拒答6次；工程错误0次。
## 运行统计
- 工程错误率：0/30（0.00%）。
- QA耗时：P50=1.413s；P95=1.759s。
- 有效耗时：30/30条；缺失或无效0条。
- `prompt_tokens`：已知合计23975；已知30/30条，缺失或无效0条。
- `completion_tokens`：已知合计1770；已知30/30条，缺失或无效0条。
- `total_tokens`：已知合计25745；已知30/30条，缺失或无效0条。
QA计时含失败尝试，不含此前检索和HTTP传输；P50为中位数，P95为95分位估计，非压测结果。缺失用量不按0计算，Token数不等于费用。

## 失败详情
没有已确认的任务失败或工程错误；未复核记录不算通过。

说明：n是各指标实际计分题数；未单列不等于0；拒答比例不等于拒答合理性。分数仅描述该次运行与该题集，不代表泛化能力。
