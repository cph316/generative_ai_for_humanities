"""爬取櫃買中心半導體產業鏈（D000）各步驟台灣公司，輸出 TXT。"""

from __future__ import annotations

import argparse
import re
from collections import OrderedDict
from html import unescape
from pathlib import Path
from typing import Dict, List, Tuple

import requests

BASE_URL = "https://ic.tpex.org.tw/introduce.php"
UPSTREAM_KEYWORDS = ("材料", "基板", "矽晶圓", "設備", "化學", "氣體", "光罩")
MIDSTREAM_KEYWORDS = ("設計", "晶圓", "製造", "製程", "代工", "IP", "IC")
DOWNSTREAM_KEYWORDS = ("封裝", "測試", "模組", "通路", "組裝", "終端")


def fetch_html(ic_code: str, timeout: int = 30) -> str:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        )
    }
    response = requests.get(BASE_URL, params={"ic": ic_code}, headers=headers, timeout=timeout)
    response.raise_for_status()
    response.encoding = response.apparent_encoding
    return response.text


def clean_text(text: str) -> str:
    text = unescape(text)
    text = re.sub(r"<[^>]+>", "", text)
    return re.sub(r"\s+", " ", text).strip()


def infer_chain_level(stage_name: str) -> str:
    if any(k in stage_name for k in UPSTREAM_KEYWORDS):
        return "上游"
    if any(k in stage_name for k in DOWNSTREAM_KEYWORDS):
        return "下游"
    if any(k in stage_name for k in MIDSTREAM_KEYWORDS):
        return "中游"
    return "未分類"


def _extract_companies_with_categories(html_fragment: str) -> List[Tuple[str, str]]:
    category_iter = list(
        re.finditer(
            r"(本國上市公司|本國上櫃公司|本國興櫃公司|本國公發公司|知名外國企業)\s*\(\d+家\)",
            html_fragment,
            re.I,
        )
    )
    rows: List[Tuple[str, str]] = []

    if category_iter:
        for i, m in enumerate(category_iter):
            category = clean_text(m.group(1))
            seg_start = m.end()
            seg_end = category_iter[i + 1].start() if i + 1 < len(category_iter) else len(html_fragment)
            segment = html_fragment[seg_start:seg_end]
            for anchor in re.findall(
                r"<a[^>]*href=['\"][^'\"]*company_basic\.php\?stk_code=\d+[^'\"]*['\"][^>]*>(.*?)</a>",
                segment,
                re.I | re.S,
            ):
                company = clean_text(anchor)
                if company and "外國" not in category:
                    rows.append((category, company))
    else:
        for anchor in re.findall(
            r"<a[^>]*href=['\"][^'\"]*company_basic\.php\?stk_code=\d+[^'\"]*['\"][^>]*>(.*?)</a>",
            html_fragment,
            re.I | re.S,
        ):
            company = clean_text(anchor)
            if company:
                rows.append(("未標註類別", company))

    return list(OrderedDict.fromkeys(rows))


def _extract_two_layer_rows(panel_html: str) -> List[Tuple[str, str]]:
    """解析兩層結構：子分類(例如 LED驅動IC) -> 公司分類 -> 公司。"""
    marker_pattern = re.compile(r"(?:^|>|\n|\r|&#9658;|▶|►)\s*([^<>]{2,40}?)\s*\(\d+家\)", re.I)
    markers = list(marker_pattern.finditer(panel_html))
    if not markers:
        return []

    out: List[Tuple[str, str]] = []
    for i, m in enumerate(markers):
        substep = clean_text(m.group(1)).lstrip("▶► ").strip()
        if (substep.startswith("本國") or "外國企業" in substep or "上市公司" in substep or "上櫃公司" in substep or "興櫃公司" in substep or "公發公司" in substep):
            continue

        start = m.end()
        end = markers[i + 1].start() if i + 1 < len(markers) else len(panel_html)
        segment = panel_html[start:end]
        rows = _extract_companies_with_categories(segment)
        for category, company in rows:
            out.append((f"{substep}｜{category}", company))

    return list(OrderedDict.fromkeys(out))


def parse_popup_sections(html: str) -> Dict[str, List[Tuple[str, str]]]:
    out: Dict[str, List[Tuple[str, str]]] = OrderedDict()

    strict_pattern = re.compile(
        r"<div[^>]*class=['\"][^'\"]*ui-dialog-titlebar[^'\"]*['\"][^>]*>"
        r".*?<span[^>]*class=['\"][^'\"]*ui-dialog-title[^'\"]*['\"][^>]*>(.*?)</span>"
        r".*?<div[^>]*class=['\"][^'\"]*company-list[^'\"]*['\"][^>]*>(.*?)</div>",
        re.I | re.S,
    )

    for title_html, list_html in strict_pattern.findall(html):
        stage_name = clean_text(title_html)
        if not stage_name:
            continue

        # 先嘗試兩層，再回退到一層
        rows = _extract_two_layer_rows(list_html)
        if not rows:
            rows = _extract_companies_with_categories(list_html)
        if rows:
            out[stage_name] = rows

    if out:
        return out

    # fallback: 針對半導體頁面常見 subchain-company-list 區塊
    panel_pattern = re.compile(
        r"<div[^>]*id=['\"]sc-ind-pnl_[^'\"]+['\"][^>]*>(.*?)</div>\s*</div>",
        re.I | re.S,
    )
    for idx, m in enumerate(panel_pattern.finditer(html), start=1):
        panel_html = m.group(1)
        rows = _extract_two_layer_rows(panel_html)
        if not rows:
            rows = _extract_companies_with_categories(panel_html)
        if not rows:
            continue

        prefix = html[max(0, m.start() - 3000):m.start()]
        title_match = re.findall(r"<h\d[^>]*>(.*?)</h\d>|<span[^>]*>([^<>]{2,30})</span>", prefix, re.I | re.S)
        stage_name = clean_text((title_match[-1][0] or title_match[-1][1])) if title_match else f"未命名步驟_{idx}"
        out[stage_name] = rows

    return out


def save_to_txt(data: Dict[str, List[Tuple[str, str]]], output_path: Path) -> None:
    lines: List[str] = []
    for stage, rows in data.items():
        lines.append(f"[{infer_chain_level(stage)}] {stage}")
        grouped: Dict[str, List[str]] = OrderedDict()
        for tag, company in rows:
            grouped.setdefault(tag, []).append(company)

        for tag, companies in grouped.items():
            lines.append(f"  ({tag})")
            for c in sorted(set(companies)):
                lines.append(f"  - {c}")
        lines.append("")

    output_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="抓取半導體產業鏈步驟中的台灣公司並輸出 TXT")
    parser.add_argument("--ic", default="D000", help="產業代碼，半導體固定使用 D000")
    parser.add_argument("--output", default="tw_semiconductor_supply_chain_companies.txt", help="輸出 txt 檔名")
    args = parser.parse_args()

    if args.ic != "D000":
        raise ValueError("目前此腳本只支援半導體產業鏈，請使用 --ic D000")

    html = fetch_html(args.ic)
    data = parse_popup_sections(html)
    if not data:
        raise RuntimeError("沒有抓到任何公司資料，請檢查網站結構或連線是否正常。")

    save_to_txt(data, Path(args.output))
    print(f"完成：共 {len(data)} 個步驟，已輸出到 {args.output}")


if __name__ == "__main__":
    main()
