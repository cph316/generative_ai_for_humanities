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

## 台灣財報 Graph RAG 問答系統（新增）

本專案新增 `tw_financial_graph_rag.py`，提供一個可直接執行的 **Graph RAG 財報問答完整程式**（CLI 版），包含：

- 知識圖譜資料模型（公司、報告、指標、事件、來源段落）
- 圖譜檢索（依公司/期間/指標查詢）
- 向量檢索（內建 TF-IDF + cosine，相容無額外套件）
- Query 解析（公司、季度、指標、比較意圖）
- 回答組裝（結論、同比、可能原因、來源引用、產生時間）

### 執行方式

```bash
python tw_financial_graph_rag.py
```

### 範例問題

- `台積電 2025Q4 毛利率較 2024Q4 變動多少？主要原因是什麼？`
- `台積電 2025Q4 EPS 是多少？`
- `2330 2024Q4 營收`

> 目前內建示範資料（demo data），可依 `load_demo_graph()` 替換為真實 ETL 管線輸入。

## 原始 Space 設定

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference
