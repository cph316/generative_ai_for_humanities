"""台灣財報問答系統（Graph RAG）範例實作。

功能：
1) 載入示範財報資料（可替換成真實 ETL）
2) 建立知識圖譜（公司、報告、指標值、事件、來源段落）
3) 建立簡易向量檢索（TF-IDF + cosine，相容無外部依賴環境）
4) 問答流程：Query 分析 -> Graph 檢索 -> Vector 補證據 -> 生成答案

執行：
    python tw_financial_graph_rag.py
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import math
import re
from typing import Dict, List, Optional, Tuple


# -----------------------
# Data Model
# -----------------------

@dataclass
class SourceChunk:
    chunk_id: str
    company: str
    period: str
    page: int
    text: str
    url: str


@dataclass
class MetricValue:
    metric: str
    value: float
    unit: str
    currency: str
    period: str
    source_chunk_id: str


@dataclass
class Event:
    title: str
    date: str
    description: str
    related_metrics: List[str] = field(default_factory=list)


@dataclass
class Report:
    period: str
    report_type: str
    publish_date: str
    metrics: List[MetricValue] = field(default_factory=list)
    chunks: List[SourceChunk] = field(default_factory=list)


@dataclass
class Company:
    ticker: str
    name: str
    industry: str
    market: str
    reports: Dict[str, Report] = field(default_factory=dict)
    events: List[Event] = field(default_factory=list)


# -----------------------
# Knowledge Graph (in-memory)
# -----------------------

class FinancialGraph:
    def __init__(self) -> None:
        self.companies: Dict[str, Company] = {}

    def upsert_company(self, company: Company) -> None:
        self.companies[company.ticker] = company

    def get_company(self, keyword: str) -> Optional[Company]:
        keyword = keyword.strip().lower()
        for c in self.companies.values():
            if keyword in c.ticker.lower() or keyword in c.name.lower():
                return c
        return None

    def get_metric_value(self, ticker: str, period: str, metric: str) -> Optional[MetricValue]:
        company = self.companies.get(ticker)
        if not company:
            return None
        report = company.reports.get(period)
        if not report:
            return None
        for mv in report.metrics:
            if mv.metric == metric:
                return mv
        return None

    def get_periods(self, ticker: str) -> List[str]:
        company = self.companies.get(ticker)
        if not company:
            return []
        return sorted(company.reports.keys())

    def get_related_events(self, ticker: str, metric: str) -> List[Event]:
        company = self.companies.get(ticker)
        if not company:
            return []
        return [e for e in company.events if metric in e.related_metrics]


# -----------------------
# Vector Retriever (TF-IDF)
# -----------------------

class TfidfRetriever:
    def __init__(self) -> None:
        self.docs: List[SourceChunk] = []
        self.vocab: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}
        self.doc_vectors: List[Dict[str, float]] = []

    @staticmethod
    def tokenize(text: str) -> List[str]:
        text = text.lower()
        text = re.sub(r"[^\w\u4e00-\u9fff]+", " ", text)
        return [t for t in text.split() if t]

    def fit(self, chunks: List[SourceChunk]) -> None:
        self.docs = chunks
        tokenized_docs = [self.tokenize(c.text) for c in chunks]

        df: Dict[str, int] = {}
        for tokens in tokenized_docs:
            for t in set(tokens):
                df[t] = df.get(t, 0) + 1

        n_docs = max(1, len(tokenized_docs))
        self.idf = {t: math.log((n_docs + 1) / (freq + 1)) + 1 for t, freq in df.items()}

        self.doc_vectors = []
        for tokens in tokenized_docs:
            tf: Dict[str, int] = {}
            for t in tokens:
                tf[t] = tf.get(t, 0) + 1
            length = max(1, len(tokens))
            vec = {t: (cnt / length) * self.idf.get(t, 0.0) for t, cnt in tf.items()}
            self.doc_vectors.append(vec)

    def _cosine(self, a: Dict[str, float], b: Dict[str, float]) -> float:
        common = set(a) & set(b)
        dot = sum(a[t] * b[t] for t in common)
        na = math.sqrt(sum(v * v for v in a.values()))
        nb = math.sqrt(sum(v * v for v in b.values()))
        if na == 0 or nb == 0:
            return 0.0
        return dot / (na * nb)

    def search(self, query: str, top_k: int = 3) -> List[Tuple[SourceChunk, float]]:
        tokens = self.tokenize(query)
        tf: Dict[str, int] = {}
        for t in tokens:
            tf[t] = tf.get(t, 0) + 1
        length = max(1, len(tokens))
        qv = {t: (cnt / length) * self.idf.get(t, 0.0) for t, cnt in tf.items()}

        scored = []
        for doc, dv in zip(self.docs, self.doc_vectors):
            scored.append((doc, self._cosine(qv, dv)))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]


# -----------------------
# Graph RAG QA Engine
# -----------------------

class GraphRAGEngine:
    METRIC_ALIASES = {
        "營收": "revenue",
        "營業收入": "revenue",
        "毛利率": "gross_margin",
        "eps": "eps",
        "每股盈餘": "eps",
        "自由現金流": "fcf",
    }

    def __init__(self, graph: FinancialGraph, retriever: TfidfRetriever) -> None:
        self.graph = graph
        self.retriever = retriever

    def parse_query(self, query: str) -> Dict[str, Optional[str]]:
        company = None
        for c in self.graph.companies.values():
            if c.name in query or c.ticker in query:
                company = c.ticker
                break

        period_match = re.findall(r"20\d{2}Q[1-4]", query)
        period = period_match[0] if period_match else None

        metric = None
        for zh, key in self.METRIC_ALIASES.items():
            if zh.lower() in query.lower():
                metric = key
                break

        compare = "較" in query or "比" in query or "變動" in query
        return {"company": company, "period": period, "metric": metric, "compare": str(compare)}

    def answer(self, query: str) -> str:
        parsed = self.parse_query(query)
        ticker = parsed["company"]
        metric = parsed["metric"]
        period = parsed["period"]
        compare = parsed["compare"] == "True"

        if not ticker:
            return "找不到公司，請輸入公司名稱或代號（例如：台積電/2330）。"
        if not metric:
            return "找不到財務指標，請明確指定（例如：營收、毛利率、EPS、自由現金流）。"

        company = self.graph.companies[ticker]

        if not period:
            periods = self.graph.get_periods(ticker)
            if not periods:
                return f"{company.name} 尚無可用報告資料。"
            period = periods[-1]

        mv = self.graph.get_metric_value(ticker, period, metric)
        if not mv:
            return f"{company.name} 在 {period} 找不到 {metric} 資料。"

        lines = [f"【結論】{company.name}（{ticker}）{period} 的 {metric} = {mv.value}{mv.unit}。"]

        if compare:
            prev_period = self._previous_same_quarter(period)
            if prev_period:
                prev = self.graph.get_metric_value(ticker, prev_period, metric)
                if prev:
                    delta = mv.value - prev.value
                    pct = (delta / prev.value * 100) if prev.value != 0 else 0.0
                    lines.append(
                        f"【同比】相較 {prev_period}，變動 {delta:.2f}{mv.unit}（{pct:.2f}%）。"
                    )

        events = self.graph.get_related_events(ticker, metric)
        if events:
            latest_event = sorted(events, key=lambda e: e.date, reverse=True)[0]
            lines.append(f"【可能原因】{latest_event.date}：{latest_event.title}。{latest_event.description}")

        evidence = self.retriever.search(f"{company.name} {period} {metric} {query}", top_k=2)
        if evidence:
            lines.append("【引用來源】")
            for doc, score in evidence:
                lines.append(
                    f"- {doc.company} {doc.period} p.{doc.page}（score={score:.3f}）{doc.url}"
                )

        lines.append(f"【產生時間】{datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
        return "\n".join(lines)

    @staticmethod
    def _previous_same_quarter(period: str) -> Optional[str]:
        m = re.match(r"(20\d{2})Q([1-4])", period)
        if not m:
            return None
        year = int(m.group(1)) - 1
        q = m.group(2)
        return f"{year}Q{q}"


# -----------------------
# Demo Data Loader
# -----------------------

def load_demo_graph() -> Tuple[FinancialGraph, TfidfRetriever]:
    graph = FinancialGraph()

    tsmc = Company(ticker="2330", name="台積電", industry="半導體", market="TWSE")

    report_2024q4 = Report(period="2024Q4", report_type="季報", publish_date="2025-01-15")
    report_2024q4.metrics.extend(
        [
            MetricValue("gross_margin", 53.1, "%", "TWD", "2024Q4", "c1"),
            MetricValue("revenue", 6250, "億元", "TWD", "2024Q4", "c2"),
            MetricValue("eps", 12.4, "元", "TWD", "2024Q4", "c3"),
        ]
    )
    report_2024q4.chunks.extend(
        [
            SourceChunk("c1", "台積電", "2024Q4", 18, "毛利率為 53.1%，受惠先進製程需求成長。", "https://example.com/tsmc_2024q4"),
            SourceChunk("c2", "台積電", "2024Q4", 9, "第四季營收 6250 億元，年增 20%。", "https://example.com/tsmc_2024q4"),
            SourceChunk("c3", "台積電", "2024Q4", 22, "每股盈餘 EPS 為 12.4 元。", "https://example.com/tsmc_2024q4"),
        ]
    )

    report_2025q4 = Report(period="2025Q4", report_type="季報", publish_date="2026-01-16")
    report_2025q4.metrics.extend(
        [
            MetricValue("gross_margin", 55.0, "%", "TWD", "2025Q4", "c4"),
            MetricValue("revenue", 7010, "億元", "TWD", "2025Q4", "c5"),
            MetricValue("eps", 13.8, "元", "TWD", "2025Q4", "c6"),
        ]
    )
    report_2025q4.chunks.extend(
        [
            SourceChunk("c4", "台積電", "2025Q4", 20, "毛利率提升至 55.0%，主因高效能運算需求。", "https://example.com/tsmc_2025q4"),
            SourceChunk("c5", "台積電", "2025Q4", 10, "第四季營收 7010 億元，持續創高。", "https://example.com/tsmc_2025q4"),
            SourceChunk("c6", "台積電", "2025Q4", 25, "每股盈餘 EPS 13.8 元。", "https://example.com/tsmc_2025q4"),
        ]
    )

    tsmc.reports[report_2024q4.period] = report_2024q4
    tsmc.reports[report_2025q4.period] = report_2025q4
    tsmc.events.append(
        Event(
            title="先進封裝產能擴充",
            date="2025-11-10",
            description="高毛利產品組合增加，帶動整體毛利率改善。",
            related_metrics=["gross_margin", "revenue"],
        )
    )

    graph.upsert_company(tsmc)

    retriever = TfidfRetriever()
    all_chunks: List[SourceChunk] = []
    for comp in graph.companies.values():
        for rpt in comp.reports.values():
            all_chunks.extend(rpt.chunks)
    retriever.fit(all_chunks)

    return graph, retriever


def main() -> None:
    graph, retriever = load_demo_graph()
    engine = GraphRAGEngine(graph, retriever)

    print("台灣財報 Graph RAG 問答系統（輸入 exit 離開）")
    while True:
        query = input("\n請輸入問題：").strip()
        if query.lower() in {"exit", "quit"}:
            print("已結束。")
            break
        print("\n" + engine.answer(query))


if __name__ == "__main__":
    main()
