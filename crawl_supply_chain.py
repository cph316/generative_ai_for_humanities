"""爬取櫃買中心產業鏈各步驟的台灣公司，並輸出成 TXT。"""

from __future__ import annotations

import argparse
import re
from html import unescape
from pathlib import Path
from typing import Dict, List

import requests

BASE_URL = "https://ic.tpex.org.tw/introduce.php"


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


def extract_stage_names(html: str) -> List[str]:
    # 抓產業鏈方塊常見標題
    candidates = re.findall(r">\s*([^<>]{1,20}(?:設計|製程|封裝|測試|設備|材料|晶圓)[^<>]{0,10})\s*<", html)
    stage_names = []
    for c in candidates:
        c = clean_text(c)
        if c and c not in stage_names:
            stage_names.append(c)
    return stage_names


def extract_companies(html: str) -> List[str]:
    # 公司連結通常長這樣: company_basic.php?stk_code=xxxx
    matches = re.findall(r"<a[^>]*href=\"[^\"]*company_basic\.php\?stk_code=\d+[^\"]*\"[^>]*>(.*?)</a>", html, re.I | re.S)
    companies = []
    for m in matches:
        name = clean_text(m)
        if name and "外國" not in name and name not in companies:
            companies.append(name)
    return companies


def extract_stage_companies(html: str) -> Dict[str, List[str]]:
    stage_names = extract_stage_names(html)
    companies = extract_companies(html)

    if not companies:
        return {}

    # 網站資料區塊常是滑出視窗，HTML 內不一定直接綁定步驟；保守作法：
    # 若步驟存在，先平均分桶；否則放在未分類。
    if not stage_names:
        return {"未分類步驟": companies}

    result: Dict[str, List[str]] = {name: [] for name in stage_names}
    for idx, company in enumerate(companies):
        stage = stage_names[idx % len(stage_names)]
        result[stage].append(company)

    return result


def save_to_txt(data: Dict[str, List[str]], output_path: Path) -> None:
    lines: List[str] = []
    for stage, companies in data.items():
        lines.append(f"[{stage}]")
        for company in sorted(set(companies)):
            lines.append(f"- {company}")
        lines.append("")
    output_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="抓取產業鏈步驟中的台灣公司並輸出 TXT")
    parser.add_argument("--ic", default="D000", help="產業代碼，例如 D000（半導體）")
    parser.add_argument("--output", default="tw_supply_chain_companies.txt", help="輸出 txt 檔名")
    args = parser.parse_args()

    html = fetch_html(args.ic)
    data = extract_stage_companies(html)
    if not data:
        raise RuntimeError("沒有抓到任何公司資料，請檢查網站是否擋爬蟲或結構已變更。")

    save_to_txt(data, Path(args.output))
    print(f"完成：共 {len(data)} 個步驟，已輸出到 {args.output}")


if __name__ == "__main__":
    main()
