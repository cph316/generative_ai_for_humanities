# 台灣財報問答系統（Graph RAG）架構設計

## 1. 目標與範圍
- **目標**：建立一個可回答「台灣上市櫃公司財報」問題的問答系統，並具備來源可追溯、指標可計算、跨公司/跨年度比較能力。
- **主要資料**：公開資訊觀測站（MOPS）年報/季報、重大訊息、公司基本資料、產業分類、IFRS 附註。
- **核心方法**：以 **Graph RAG（Knowledge Graph + Vector RAG）** 結合精準檢索與推理。

## 2. 為什麼用 Graph RAG
傳統 RAG 對單一段落查找很強，但在財報場景常遇到：
1. 同一個概念散落在多頁（例如「存貨跌價損失」在附註）。
2. 問題需要關聯多表（損益表 + 現金流量表 + 公司事件）。
3. 使用者常問「比較」與「因果」（如毛利率變動主因）。

Graph RAG 透過「實體-關係」結構，先在圖譜做結構化召回，再回到文本證據完成答案。

## 3. 高階架構

```text
[Data Sources]
  ├─ MOPS 財報 PDF/HTML
  ├─ 公司基本資料/產業分類
  └─ 重大訊息與法說會資料

        ↓ ETL + Parsing

[Document Store]
  ├─ 原始檔 (S3/MinIO)
  └─ Chunked text + metadata (Postgres/Elastic)

        ↓ IE/NER/RE + Table Parsing

[Knowledge Graph]
  ├─ Entity: 公司、期間、科目、金額、比率、事件
  ├─ Relation: belongs_to, reported_in, yoy_change, caused_by
  └─ Graph DB: Neo4j/TigerGraph

        ↓ Hybrid Retrieval

[Retriever Layer]
  ├─ Graph query (Cypher)
  ├─ Vector search (Milvus/pgvector)
  └─ Re-ranker (cross-encoder)

        ↓

[LLM Orchestrator]
  ├─ Query decomposition
  ├─ Tool routing (graph/vector/sql)
  ├─ Evidence grounding + citation
  └─ Response synthesis (繁中)

        ↓

[Application Layer]
  ├─ Web Chat
  ├─ Analyst mode（比較、趨勢、預警）
  └─ API（內部系統整合）
```

## 4. 知識圖譜 Schema（建議）

### 4.1 節點（Nodes）
- `Company`：公司代號、公司名稱、產業別、市場別。
- `Report`：年/季、財報種類、發布日期、審計狀態。
- `Metric`：指標名稱（營收、毛利、EPS、自由現金流）。
- `MetricValue`：數值、幣別、單位、期間。
- `Event`：重大訊息（增資、訴訟、停工、併購）。
- `Note`：附註段落與風險揭露。
- `SourceChunk`：對應回原文段落位置（頁碼、段落 id、URL）。

### 4.2 關係（Edges）
- `(Company)-[:ISSUED]->(Report)`
- `(Report)-[:HAS_METRIC]->(MetricValue)`
- `(MetricValue)-[:OF_METRIC]->(Metric)`
- `(MetricValue)-[:SUPPORTED_BY]->(SourceChunk)`
- `(Event)-[:AFFECTS]->(MetricValue)`
- `(MetricValue)-[:YOY_CHANGE]->(MetricValue)`
- `(Report)-[:HAS_NOTE]->(Note)`

### 4.3 索引
- 圖索引：`Company.ticker`、`Report.period`、`Metric.name`。
- 向量索引：`SourceChunk.embedding`。
- 關鍵字索引：財務科目同義詞（例：營業利益 = 營業淨利）。

## 5. 資料處理流程（ETL）
1. **擷取**：定期抓取 MOPS 新增財報與更正公告。
2. **解析**：
   - PDF/HTML 轉文字。
   - 表格抽取（保留列欄語意）。
   - 單位標準化（千元/百萬元、幣別）。
3. **資訊抽取**：
   - NER：公司、科目、期間、數值。
   - RE：科目與期間對應、數值與來源對應。
4. **圖譜寫入**：Upsert nodes/edges，避免重複。
5. **向量化**：對附註、MD&A、重大訊息 chunk 建 embedding。
6. **品質檢核**：
   - 平衡檢查（資產=負債+權益）。
   - 同期一致性檢查（Q4 累計 vs 年報）。

## 6. 查詢流程（Inference）
1. 使用者問題進入 `Query Analyzer`。
2. 分解問題（例：公司 + 指標 + 期間 + 比較條件）。
3. 先做 Graph retrieval：抓候選實體與關係路徑。
4. 再做 Vector retrieval：補齊附註文字與管理層說明。
5. Re-ranker 依問題重排證據。
6. LLM 生成答案，輸出：
   - 結論
   - 關鍵數據表
   - 變動原因
   - 引用來源（頁碼/段落）
   - 信心分數

## 7. 提示詞（Prompt）策略
- **System Prompt**：限定只可依證據回答，禁止臆測。
- **Planner Prompt**：先規劃需要的子查詢（圖譜/向量/計算）。
- **Answer Prompt**：統一輸出格式（摘要、表格、引用、風險）。
- **Guardrail Prompt**：若證據不足，明確回答「資料不足」。

## 8. 評估指標
- **Retrieval**：Recall@k、MRR（圖譜與向量分開看）。
- **Answer**：
  - Faithfulness（是否被證據支持）
  - Numeric Accuracy（數值誤差率）
  - Citation Precision（引用是否正確對位）
- **Business KPI**：分析師查詢時間縮短、覆盤報告產出速度。

## 9. 技術選型（台灣財報場景建議）
- **Ingestion**：Airflow / Dagster。
- **Parsing**：pdfplumber + camelot/tabula + 自訂欄位清洗。
- **Graph DB**：Neo4j（快啟動、Cypher 生態成熟）。
- **Vector DB**：pgvector（若已用 Postgres）或 Milvus（大規模）。
- **LLM**：可替換式（OpenAI / 本地模型），建議繁中強模型。
- **Serving**：FastAPI + Streamlit/React。

## 10. MVP 落地路線（8–10 週）
- **第 1–2 週**：資料範圍定義（先 50 家公司、近 3 年季報）。
- **第 3–4 週**：ETL + 表格抽取 + metadata schema。
- **第 5–6 週**：建圖 + 向量檢索 + hybrid retriever。
- **第 7–8 週**：問答鏈路 + 引用 + 評估集。
- **第 9–10 週**：優化（同義詞、數值校驗、查詢快取）。

## 11. 典型問題範例（系統可回答）
1. 「台積電 2025Q4 毛利率較 2024Q4 變動多少？主要原因是什麼？」
2. 「聯發科近 8 季存貨週轉天數趨勢，異常點對應哪些重大訊息？」
3. 「面板產業中，最近一年自由現金流轉負的公司有哪些？附依據。」

## 12. 風險與治理
- **法規遵循**：僅使用公開資訊，保留來源鏈結與抓取時間。
- **模型幻覺**：強制 citation + 數值交叉檢查。
- **更新延遲**：建立增量更新與失敗重跑機制。
- **可稽核性**：保留查詢日志與版本化 prompt。

---

## 建議下一步
- 先做一個「單公司 + 8 季 + 三大表 + 附註」的 PoC，驗證：
  1) 數值正確率、2) 引用正確率、3) 使用者主觀可用性。
