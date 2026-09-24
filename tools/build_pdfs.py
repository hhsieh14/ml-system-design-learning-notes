"""Rebuild chapter PDFs and the combined handbook from the Markdown sources.

Usage (from the repository root):  python tools/build_pdfs.py
Needs pandoc and xelatex. GitHub-only syntax (alerts, <details>) is converted
to plain Markdown first so the PDF shows the same content.
"""
import re, subprocess, tempfile, pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
CHAPTERS = [
    "00_framework/chapter_00_ml_system_design_framework",
    "01_ranking_model/chapter_01_design_a_ranking_model",
    "02_ad_click_prediction/chapter_02_ad_click_prediction",
    "03_people_you_may_know/chapter_03_people_you_may_know",
    "04_harmful_content_detection/chapter_04_harmful_content_detection",
]
PANDOC = ["pandoc", "--pdf-engine=xelatex", "-V", "fontfamily=fontspec", "-V", "geometry:margin=2.2cm", "-V", "mainfont=DejaVu Serif",
          "-V", "sansfont=DejaVu Sans", "-V", "monofont=DejaVu Sans Mono", "-V", "fontsize=10pt",
          "-V", "colorlinks=true", "-V", "linkcolor=blue", "-V", "urlcolor=blue",
          "--from", "markdown+tex_math_dollars"]


def to_print_markdown(text: str, img_prefix: str = "") -> str:
    text = re.sub(r"^> \[!(NOTE|TIP|WARNING|IMPORTANT|CAUTION)\]\s*\n", lambda m: f"> **{m.group(1).title()}.** ", text, flags=re.M)
    text = re.sub(r"<details><summary>(.*?)</summary>", r"**Q: \1**", text)
    text = text.replace("</details>", "")
    text = re.sub(r"^\[← .*$|^\[Next: .*$", "", text, flags=re.M)          # web-only navigation
    if img_prefix:
        text = re.sub(r"!\[([^\]]*)\]\((diagrams/[^)]+)\)", rf"![\1]({img_prefix}\2)", text)
    text = re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", r"![\1](\2){ width=100% }", text)
    return text


def build(md_text: str, out: pathlib.Path, resource_paths):
    with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False) as f:
        f.write(md_text)
    subprocess.run(PANDOC + [f.name, "-o", str(out), "--resource-path", ":".join(map(str, resource_paths))], check=True)


if __name__ == "__main__":
    parts = []
    for c in CHAPTERS:
        src = ROOT / "chapters" / f"{c}.md"
        folder = src.parent
        build(to_print_markdown(src.read_text()), src.with_suffix(".pdf"), [folder])
        parts.append(to_print_markdown(src.read_text(), img_prefix=f"chapters/{folder.name}/"))
        print("built", src.with_suffix(".pdf").name)
    title = "---\ntitle: Machine Learning System Design Study Handbook\nauthor: Hsiang-Yu (Andy) Hsieh\n---\n\n"
    handbook = title + "\n\n\\newpage\n\n".join(parts)
    build(handbook, ROOT / "handbook" / "ML_System_Design_Study_Handbook.pdf", [ROOT])
    print("built handbook")
