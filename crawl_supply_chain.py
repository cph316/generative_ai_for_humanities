"""爬取櫃買中心產業鏈各步驟的台灣公司，並輸出成 TXT。"""

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


def _extract_companies_with_categories(list_html: str) -> List[Tuple[str, str]]:
    category_iter = list(
        re.finditer(
            r"(本國上市公司|本國上櫃公司|本國興櫃公司|本國公發公司|知名外國企業)\s*\(\d+家\)",
            list_html,
            re.I,
        )
    )

    companies: List[Tuple[str, str]] = []
    if category_iter:
        for i, m in enumerate(category_iter):
            category = clean_text(m.group(1))
            seg_start = m.end()
            seg_end = category_iter[i + 1].start() if i + 1 < len(category_iter) else len(list_html)
            segment = list_html[seg_start:seg_end]
            for anchor in re.findall(
                r"<a[^>]*href=['\"][^'\"]*company_basic\.php\?stk_code=\d+[^'\"]*['\"][^>]*>(.*?)</a>",
                segment,
                re.I | re.S,
            ):
                name = clean_text(anchor)
                if not name or "外國" in category:
                    continue
                companies.append((category, name))
    else:
        for anchor in re.findall(
            r"<a[^>]*href=['\"][^'\"]*company_basic\.php\?stk_code=\d+[^'\"]*['\"][^>]*>(.*?)</a>",
            list_html,
            re.I | re.S,
        ):
            name = clean_text(anchor)
            if name:
                companies.append(("未標註類別", name))

    return list(OrderedDict.fromkeys(companies))


def _guess_stage_name(prefix_html: str, fallback_idx: int) -> str:
    candidates = re.findall(
        r"<span[^>]*class=['\"][^'\"]*ui-dialog-title[^'\"]*['\"][^>]*>(.*?)</span>|"
        r"<div[^>]*class=['\"][^'\"]*(?:step|title|chain)[^'\"]*['\"][^>]*>(.*?)</div>",
        prefix_html,
        re.I | re.S,
    )
    texts = []
    for a, b in candidates:
        t = clean_text(a or b)
        if 1 <= len(t) <= 80:
            texts.append(t)
    if texts:
        return texts[-1]
    return f"未命名步驟_{fallback_idx}"


def parse_popup_sections(html: str) -> Dict[str, List[Tuple[str, str]]]:
    out: Dict[str, List[Tuple[str, str]]] = OrderedDict()

    # 先嘗試精準模式（title + company-list）
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
        rows = _extract_companies_with_categories(list_html)
        if rows:
            out[stage_name] = rows

    if out:
        return out

    # fallback: 只找 company-list，再從前文猜步驟名稱
    list_pattern = re.compile(r"<div[^>]*class=['\"][^'\"]*company-list[^'\"]*['\"][^>]*>(.*?)</div>", re.I | re.S)
    for idx, m in enumerate(list_pattern.finditer(html), start=1):
        list_html = m.group(1)
        rows = _extract_companies_with_categories(list_html)
        if not rows:
            continue
        prefix = html[max(0, m.start() - 2000):m.start()]
        stage_name = _guess_stage_name(prefix, idx)
        if stage_name in out:
            out[stage_name].extend(rows)
            out[stage_name] = list(OrderedDict.fromkeys(out[stage_name]))
        else:
            out[stage_name] = rows

    if out:
        return out

    # 最後 fallback：全頁公司連結至少先輸出，避免直接 RuntimeError
    all_companies = []
    for anchor in re.findall(
        r"<a[^>]*href=['\"][^'\"]*company_basic\.php\?stk_code=\d+[^'\"]*['\"][^>]*>(.*?)</a>",
        html,
        re.I | re.S,
    ):
        name = clean_text(anchor)
        if name:
            all_companies.append(("未標註類別", name))
    dedup = list(OrderedDict.fromkeys(all_companies))
    if dedup:
        out["未分類步驟"] = dedup

    return out


def save_to_txt(data: Dict[str, List[Tuple[str, str]]], output_path: Path) -> None:
    lines: List[str] = []
    for stage, company_rows in data.items():
        chain_level = infer_chain_level(stage)
        lines.append(f"[{chain_level}] {stage}")

        grouped: Dict[str, List[str]] = OrderedDict()
        for category, company in company_rows:
            grouped.setdefault(category, []).append(company)

        for category, companies in grouped.items():
            lines.append(f"  ({category})")
            for company in sorted(set(companies)):
                lines.append(f"  - {company}")
        lines.append("")

    output_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="抓取產業鏈步驟中的台灣公司並輸出 TXT")
    parser.add_argument("--ic", default="D000", help="產業代碼，例如 D000（半導體）")
    parser.add_argument("--output", default="tw_supply_chain_companies.txt", help="輸出 txt 檔名")
    args = parser.parse_args()

    html = fetch_html(args.ic)
    data = parse_popup_sections(html)
    if not data:
        raise RuntimeError("沒有抓到任何公司資料，請檢查網站是否擋爬蟲或結構已變更。")

    save_to_txt(data, Path(args.output))
    print(f"完成：共 {len(data)} 個步驟，已輸出到 {args.output}")


if __name__ == "__main__":
    main()
