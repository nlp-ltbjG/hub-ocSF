import re
import json
import logging
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional

import requests
from bs4 import BeautifulSoup

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

RAW_DIR = Path(__file__).parent.parent / "data" / "raw_html"
PARSED_DIR = Path(__file__).parent.parent / "data" / "parsed"
PARSED_DIR.mkdir(parents=True, exist_ok=True)


# ── 数据结构 ──────────────────────────────────────────────────────────────────


@dataclass
class ParsedBlock:
    block_type: str  # "text" | "table" | "title"
    content: str  # 文字内容（表格转为 markdown）
    page_num: int
    section_path: list[str]  # ["第三章 管理层讨论", "一、经营情况概述"]
    is_ocr: bool = False
    raw_table: Optional[list] = field(default=None, repr=False)


# ── 工具函数 ──────────────────────────────────────────────────────────────────

CHAPTER_PATTERNS = [
    re.compile(r"^第[一二三四五六七八九十百]+[章节]\s*"),
    re.compile(r"^[一二三四五六七八九十]、"),
    re.compile(r"^\d+\.\d+(\.\d+)?\s*"),
    re.compile(r"^\d+\.\s*"),
]

NOISE_WORDS = {
    "search",
    "quick search",
    "code",
    "show source",
    "source",
    "mxnet",
    "pytorch",
    "jupyter",
    "notebook",
    "记事本",
    "github",
    "english",
    "中文",
    "课程",
    "table of contents",
    "keyboard_arrow_down",
    "keyboard_arrow_right",
    "navigation",
    "nav",
    "menu",
    "toc",
    "目录",
    "首页",
    "上一页",
    "下一页",
    "返回",
    "前进",
    "登录",
    "注册",
    "帮助",
    "关于",
}

NOISE_PATTERNS = [
    re.compile(r"^.{1,40}\s*$"),
    re.compile(r"^\d+\s*$"),
    re.compile(r"^—\s*\d+\s*—$"),
]


def is_noise_line(line: str) -> bool:
    line = line.strip().lower()
    if len(line) < 2:
        return True
    if line in NOISE_WORDS:
        return True
    return any(p.match(line) for p in NOISE_PATTERNS)


def is_title_line(line: str) -> bool:
    return any(p.match(line.strip()) for p in CHAPTER_PATTERNS)


def table_to_markdown(table: list[list]) -> str:
    if not table:
        return ""

    rows = []
    for row in table:
        cleaned = [str(cell or "").replace("\n", " ").strip() for cell in row]
        rows.append(cleaned)

    if not rows:
        return ""

    header = rows[0]
    lines = ["| " + " | ".join(header) + " |"]
    lines.append("| " + " | ".join(["---"] * len(header)) + " |")
    for row in rows[1:]:
        while len(row) < len(header):
            row.append("")
        lines.append("| " + " | ".join(row[: len(header)]) + " |")

    return "\n".join(lines)


def fetch_html(source: str) -> str:
    if source.startswith(("http://", "https://")):
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        }
        resp = requests.get(source, headers=headers, timeout=30)
        resp.raise_for_status()
        return resp.text
    else:
        with open(source, "r", encoding="utf-8") as f:
            return f.read()


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    lines = []
    for line in text.split("\n"):
        line = line.strip()
        line = line.rstrip("¶")

        if not line:
            continue
        if line.lower() in NOISE_WORDS:
            continue
        lines.append(line)

    return "\n".join(lines)


def clean_soup(soup: BeautifulSoup) -> BeautifulSoup:
    for tag in soup.find_all(["script", "style", "noscript", "iframe"]):
        tag.decompose()

    for tag in soup.find_all(["nav", "header", "footer", "aside"]):
        tag.decompose()

    return soup


# ── 主解析逻辑 ────────────────────────────────────────────────────────────────


class AnnualReportHTMLParser:
    def __init__(self, html_source: str, meta: dict = None, is_url: bool = False):
        self.html_source = html_source
        self.meta = meta or {}
        self.blocks: list[ParsedBlock] = []
        self._section_stack: list[str] = []
        self._is_url = is_url

    def _update_section(self, title: str):
        if re.match(r"^第[一二三四五六七八九十]+章", title):
            self._section_stack = [title]
        elif re.match(r"^第[一二三四五六七八九十]+节", title):
            self._section_stack = self._section_stack[:1] + [title]
        elif re.match(r"^[一二三四五六七八九十]、", title):
            self._section_stack = self._section_stack[:2] + [title]
        elif re.match(r"^\d+\.\s", title):
            self._section_stack = self._section_stack[:3] + [title]
        else:
            self._section_stack = self._section_stack[:3] + [title]

    def parse(self) -> list[ParsedBlock]:
        logger.info(f"开始解析: {'URL' if self._is_url else self.html_source}")

        html_content = fetch_html(self.html_source)
        soup = BeautifulSoup(html_content, "html.parser")
        soup = clean_soup(soup)

        for tag in soup.find_all(
            [
                "h1",
                "h2",
                "h3",
                "h4",
                "h5",
                "h6",
                "p",
                "table",
                "div",
                "span",
                "li",
                "td",
                "th",
            ]
        ):
            if tag.name.startswith("h"):
                text = tag.get_text().strip()
                text = text.replace("\r\n", "\n").replace("\r", "\n")

                first_line = text.split("\n")[0].strip()
                first_line = first_line.rstrip("¶")

                if first_line and not is_noise_line(first_line):
                    self._update_section(first_line)
                    self.blocks.append(
                        ParsedBlock(
                            block_type="title",
                            content=first_line,
                            page_num=1,
                            section_path=list(self._section_stack),
                        )
                    )

            elif tag.name == "table":
                table_data = []
                headers = []
                for th in tag.find_all("th"):
                    headers.append(normalize_text(th.get_text()))
                if headers:
                    table_data.append(headers)

                for tr in tag.find_all("tr"):
                    row = []
                    cells = tr.find_all(["td", "th"])
                    for cell in cells:
                        row.append(normalize_text(cell.get_text()))
                    if row:
                        table_data.append(row)

                if table_data:
                    md = table_to_markdown(table_data)
                    if md:
                        self.blocks.append(
                            ParsedBlock(
                                block_type="table",
                                content=md,
                                page_num=1,
                                section_path=list(self._section_stack),
                                raw_table=table_data,
                            )
                        )

            elif tag.name in ["p", "div", "span", "li"]:
                text = normalize_text(tag.get_text())
                if text and not is_noise_line(text) and len(text) > 5:
                    lines = text.split("\n")
                    if len(lines) > 0 and is_title_line(lines[0]):
                        title_text = lines[0].strip()
                        self._update_section(title_text)
                        self.blocks.append(
                            ParsedBlock(
                                block_type="title",
                                content=title_text,
                                page_num=1,
                                section_path=list(self._section_stack),
                            )
                        )
                        if len(lines) > 1:
                            remaining = "\n".join(lines[1:]).strip()
                            if remaining and len(remaining) > 5:
                                self.blocks.append(
                                    ParsedBlock(
                                        block_type="text",
                                        content=remaining,
                                        page_num=1,
                                        section_path=list(self._section_stack),
                                    )
                                )
                    else:
                        self.blocks.append(
                            ParsedBlock(
                                block_type="text",
                                content=text,
                                page_num=1,
                                section_path=list(self._section_stack),
                            )
                        )

        self._deduplicate_blocks()

        logger.info(f"  解析完成: {len(self.blocks)} 个块")
        return self.blocks

    def _deduplicate_blocks(self):
        seen = set()
        unique = []
        for block in self.blocks:
            key = (block.block_type, block.content[:200])
            if key not in seen:
                seen.add(key)
                unique.append(block)
        self.blocks = unique

    def save(self):
        if self._is_url:
            stem = (
                self.html_source.replace("http://", "")
                .replace("https://", "")
                .replace("/", "_")[:50]
            )
        else:
            stem = Path(self.html_source).stem

        out_path = PARSED_DIR / f"{stem}.json"

        output = {
            "meta": self.meta,
            "source": self.html_source,
            "blocks": [asdict(b) for b in self.blocks],
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        logger.info(f"  已保存 → {out_path}")


# ── 主流程 ────────────────────────────────────────────────────────────────────


def main():
    manifest_path = RAW_DIR.parent / "manifest_html.json"

    if manifest_path.exists():
        with open(manifest_path, encoding="utf-8") as f:
            manifest = json.load(f)
    else:
        manifest = []
        for p in RAW_DIR.glob("*.html"):
            manifest.append(
                {
                    "filename": p.name,
                    "stock_code": "",
                    "year": "",
                    "is_url": False,
                    "source_path": str(p),
                }
            )

    if not manifest:
        logger.error("没有找到任何 HTML，请先准备 HTML 文件或配置 manifest_html.json")
        return

    for item in manifest:
        if item.get("is_url"):
            html_source = item.get("source_url", "")
        else:
            html_source = RAW_DIR / item.get("filename", "")
            if not Path(html_source).exists():
                logger.warning(f"文件不存在，跳过: {html_source}")
                continue

        parser = AnnualReportHTMLParser(
            html_source=str(html_source),
            meta=item,
            is_url=item.get("is_url", False),
        )
        parser.parse()
        parser.save()

    logger.info(f"\n全部解析完成，结果在 {PARSED_DIR}")


if __name__ == "__main__":
    main()
