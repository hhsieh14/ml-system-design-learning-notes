"""Replace one line of caption text in a rendered diagram PNG.

Usage (from the repository root):  python tools/patch_diagram_text.py
Each entry: (image, y_min, y_max, old_text, new_text). The band y_min..y_max must
contain only the caption line(s) to replace. The font size is matched by fitting
the width of the old text in DejaVu Sans, the font the diagrams were drawn with.
"""
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"

PATCHES = [
    ("chapters/01_ranking_model/diagrams/02_product_objective_labels.png", 755, 800,
     "Important: 1 and 0 mean positive and negative outcomes for the selected label. The notes do not define a third unlabeled state.",
     "Each head gets a binary label: 1 = the behavior happened inside its window, 0 = it did not."),
    ("chapters/02_ad_click_prediction/diagrams/09_continual_learning.png", 750, 795,
     "These methods may be used independently or combined; the notes do not prescribe one universal strategy.",
     "They can be used alone or combined; pick the smallest one that fixes the measured problem."),
    ("chapters/03_people_you_may_know/diagrams/02_scale_constraints.png", 115, 150,
     "The numbers are capacity-planning assumptions from the notes, not verified facts about a specific platform.",
     "Illustrative capacity-planning assumptions, not figures from any specific platform."),
    ("chapters/03_people_you_may_know/diagrams/06_temporal_gnn.png", 780, 815,
     "LSTM or GRU components are examples from the notes; the exact recurrent mechanism remains an implementation choice.",
     "The memory updater can be a GRU or LSTM cell (as in TGN); the exact recurrent mechanism is an implementation choice."),
    ("chapters/03_people_you_may_know/diagrams/07_candidate_generation.png", 695, 750,
     "The notes contain a rough, unverified FoF observation. The safe design conclusion is only that FoF is a plausible high-value candidate source, not",
     "Friends-of-friends is usually the highest-yield source; the others add coverage and cold-start candidates."),
    ("chapters/03_people_you_may_know/diagrams/11_negatives_evaluation.png", 115, 150,
     "Negative-pair sampling is an added technical extension, clearly separated from the original notes.",
     "Which non-connected pairs count as negatives shapes what the model learns."),
]


def patch(path, y0, y1, old, new):
    im = Image.open(path).convert("RGB")
    a = np.asarray(im).astype(int)
    band = a[y0:y1]
    ink = (np.abs(band - 255).sum(axis=2) > 60)
    rows, cols = np.where(ink)
    top, bottom = y0 + rows.min(), y0 + rows.max()
    left, right = cols.min(), cols.max()
    px = band[ink]
    dark = px[px.sum(axis=1) <= np.percentile(px.sum(axis=1), 15)]
    color = tuple(int(c) for c in np.median(dark, axis=0))   # core stroke colour, not anti-aliased edges
    first_line_w = right - left                   # width of the widest original line
    size = 10
    while ImageFont.truetype(FONT, size + 1).getlength(old) <= first_line_w + 2:
        size += 1
    font = ImageFont.truetype(FONT, size)
    d = ImageDraw.Draw(im)
    d.rectangle([left - 3, top - 3, right + 3, bottom + 3], fill=(255, 255, 255))
    asc = font.getbbox("T")[1]
    d.text((left, top - asc), new, font=font, fill=color)
    im.save(path, optimize=True)
    print(f"{path}: size {size}, band {top}-{bottom}")


if __name__ == "__main__":
    for p in PATCHES:
        patch(*p)
