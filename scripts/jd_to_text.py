"""Turn a fetched JD (iCIMS page with ld+json, or SmartRecruiters posting JSON) into plain text.

Usage: python3 -I jd_to_text.py icims|smartrecruiters IN_FILE OUT_FILE URL FETCHED_DATE

The raw response stays in jd/raw/ so the text can be regenerated after the posting is gone.
"""
import html
import json
import re
import sys
from html.parser import HTMLParser


class _Text(HTMLParser):
    BLOCK = {"p", "div", "br", "ul", "ol", "h1", "h2", "h3", "h4", "h5", "h6", "tr"}

    def __init__(self):
        super().__init__()
        self.out = []

    def handle_starttag(self, tag, attrs):
        if tag == "li":
            self.out.append("\n- ")
        elif tag in self.BLOCK:
            self.out.append("\n")

    def handle_endtag(self, tag):
        if tag in self.BLOCK or tag == "li":
            self.out.append("\n")

    def handle_data(self, data):
        self.out.append(data)


def to_text(fragment: str) -> str:
    p = _Text()
    p.feed(html.unescape(fragment) if "&lt;" in fragment else fragment)
    text = "".join(p.out).replace("\xa0", " ")
    lines = [re.sub(r"[ \t]+", " ", ln).strip() for ln in text.splitlines()]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def main():
    kind, src, dst, url, fetched = sys.argv[1:6]
    raw = open(src, encoding="utf-8").read()
    if kind == "icims":
        block = re.search(r'<script type="application/ld\+json">(.*?)</script>', raw, re.S).group(1)
        d = json.loads(block)
        head = f"{d.get('title')}"
        body = to_text(d["description"])
    else:
        d = json.loads(raw)
        head = f"{d.get('name')} :: {d.get('location', {}).get('fullLocation')}"
        parts = []
        for sec in d["jobAd"]["sections"].values():
            if sec.get("text"):
                parts.append(f"## {sec.get('title')}\n\n{to_text(sec['text'])}")
        body = "\n\n".join(parts)
    with open(dst, "w", encoding="utf-8") as f:
        f.write(f"# source: {url}\n# fetched: {fetched}\n# title: {head}\n\n{body}\n")


main()
