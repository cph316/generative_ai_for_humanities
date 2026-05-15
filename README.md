---
title: Travel Assistant
emoji: 👀
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: 5.33.0
app_file: app.py
pinned: false
short_description: 期末專題
---

## 台灣財報 Graph RAG 問答系統

### 目前語言模型是什麼？
- **預設沒有外接 LLM**，目前回答邏輯是 `GraphRAGEngine` 依據圖譜與檢索證據做規則化生成（不是 GPT）。
- 這樣做的好處是：可離線測試、可控、可追溯。
- 你若要接 OpenAI/Groq，可在後續把 `engine.answer()` 改成「證據檢索 + LLM 整理」。

### 如何使用真實財報資料（MOPS）
程式已支援直接抓 MOPS 季報表格：

```bash
python tw_financial_graph_rag.py --use-real-data --company-id 2330 --roc-year 114 --season 4
```

參數說明：
- `--company-id`：公司代號（例：2330）
- `--roc-year`：民國年（114 = 西元 2025）
- `--season`：季度（1~4）

> 若抓取失敗（例如網路/站台限制），程式會自動 fallback 到內建 demo 資料。

### 一般執行（demo 資料）

```bash
python tw_financial_graph_rag.py
```
