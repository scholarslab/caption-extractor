#!/usr/bin/env python3
"""Detect individual artwork photos on scanned catalog pages using OpenCV.

Pages are busy scans: photographic reproductions of paintings placed on an
off-white page, next to printed caption text. Two complementary passes are
used:

  Stage A - bordered artworks. Most reproductions have a clean rectangular
  edge (a printed rule, or just the sharp tonal jump from photo to white
  page) that Canny traces as a near-perfect rectangle contour. This is the
  precise, high-confidence pass.

  Stage B - bleeding artworks. Some reproductions run to the trim edge of
  the page with no border, so they have no closed rectangle to find. These
  are instead picked up as large, densely-textured blobs (paintings have
  edges/texture everywhere; the surrounding page and caption text don't).
  A thin strip is blanked out at the page's outer border first, since that
  border is otherwise one continuous line that bridges a bleeding artwork
  into the whole page as a single connected component. The resulting box is
  then trimmed on each side using row/column edge-density, since it commonly
  captures a trailing run of adjacent caption text.

This is a heuristic, not a segmentation model - expect some misses and some
loose boxes on busy or low-contrast pages. Use --debug-dir output to review
results visually; adjust the thresholds below (or pass CLI flags) to retune
for a different scan set.

Runs against the *_small.png previews in pages/ for speed; bounding boxes are
reported in those preview-image pixel coordinates (same convention as
image-captions/KIC.json).

---------------------------------------------------------------------------
HOW THIS WORKS - a computer-vision primer for people who haven't done any

No machine learning here; every step is a plain, deterministic image
transform. If you can picture what each of these does to a photo, you can
picture the whole pipeline:

* Grayscale + Gaussian blur (cv2.cvtColor, cv2.GaussianBlur). An image is
  just a 2D array of pixel brightnesses (0-255). We drop color (we only
  care about *shape*, not hue) and blur it very slightly so that scan
  grain/JPEG speckle doesn't get mistaken for a real edge in the next step.

* Canny edge detection (cv2.Canny). Walks the array looking for places
  where brightness changes sharply between neighboring pixels - i.e. an
  edge - and outputs a same-size array that's 255 at those pixels and 0
  everywhere else ("edges", a binary/black-and-white mask). `canny_low`/
  `canny_high` control how strong a brightness jump has to be to count.
  A photo of a painting has edges scattered densely all over it (every
  brushstroke, every tonal shift); a blank page or a gap between letters
  has almost none. That density difference is the signal this whole script
  is built on.

* Morphology: dilate and close (cv2.dilate, cv2.morphologyEx with
  MORPH_CLOSE). These operate on a binary mask using a small square
  "kernel" (e.g. 5x5 or 11x11 pixels) that slides over every pixel.
  Dilate: a pixel becomes 255 if *any* pixel under the kernel is 255 - it
  grows white regions outward, which bridges together edges that are close
  but not quite touching (e.g. Canny broke one brushstroke's outline into
  a few short dashes; dilating reconnects them into one shape). Close is
  dilate followed by erode (the opposite: shrink white regions back down);
  it fills tiny gaps/holes without permanently growing the overall shape
  much, unlike a plain dilate which keeps every mask that size. Kernel size
  controls how big a gap gets bridged - too small and a real edge stays in
  pieces, too big and unrelated things merge into one blob (this bit us
  during development: see find_bleeding_boxes' border_strip trick below).

* Contours (cv2.findContours). Once you have a binary mask, a "contour" is
  just the traced outline of a connected white blob - think of it like
  tracing a coastline on a black-and-white map. It hands back each blob as
  a polygon (a list of boundary points), from which cv2.boundingRect gives
  you the smallest upright rectangle that contains it, and cv2.contourArea
  gives you the actual enclosed area of that polygon (which is smaller than
  the bounding rectangle's area unless the blob IS a filled rectangle).

  One gotcha that cost real debugging time: findContours can return either
  every blob (RETR_LIST) or only the outermost blob of each nested group
  (RETR_EXTERNAL), discarding whatever is inside it. A scanned page has a
  faint rectangular frame around the whole photographed spread; with
  RETR_EXTERNAL that outer frame swallows every artwork inside it into one
  contour and you get a single box covering the entire page. RETR_LIST
  fixes that by returning inner blobs too, at the cost of also returning
  some we don't want (duplicate/nested tracings of the same border, at
  slightly different offsets) - hence the fill-ratio and dedupe logic below.

* "Fill ratio" (contourArea / boundingRect area). A contour that traces a
  clean filled rectangle has a polygon area almost equal to its own
  bounding box's area (ratio close to 1.0). A contour tracing a blob of
  caption text, or a ragged/irregular shape, encloses a lot of empty space
  relative to its bounding box (ratio well below 1.0). This one number is
  what separates "this is a rectangle" from "this is not," and is the core
  of Stage A.

* Edge density (cv2.countNonZero(region) / region area). Same idea applied
  differently: what fraction of pixels *inside a candidate box* are edge
  pixels at all? A photo is densely textured everywhere, so this stays
  high across its whole area; blank margin or sparse caption text pulls it
  down. This is the core of Stage B, used instead of fill ratio because a
  bleeding artwork's contour isn't a clean rectangle to begin with.

* IoU, "Intersection over Union" (iou_over_union/iou_over_smaller below).
  The standard way to measure how much two boxes overlap: the area they
  share, divided by either their combined area (union) or the smaller box's
  own area. It's just arithmetic on rectangles, used here purely to detect
  "these two candidate boxes are actually the same artwork found twice."
---------------------------------------------------------------------------
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

PAGES_DIR = Path("pages")
OUT_JSON = Path("image-captions/detected.json")
DEBUG_DIR = Path("image-captions/detected_debug")

Box = tuple[int, int, int, int]  # xmin, ymin, xmax, ymax


# --- Box overlap (IoU) helpers -------------------------------------------
#
# Both compute the same intersection rectangle (max of the two left/top
# edges to min of the two right/bottom edges - if that comes out inverted,
# the boxes don't overlap at all, hence the early 0.0 return). They only
# differ in what they divide by, which changes what a "high score" means:


def iou_over_smaller(a: Box, b: Box) -> float:
    """Intersection as a fraction of the SMALLER box's area.

    Answers "is the smaller box basically sitting inside the bigger one?" -
    1.0 means it's fully contained, regardless of how much larger the other
    box is. Used where containment itself is the thing we care about (e.g.
    "did Stage B just re-find a Stage A box, plus some extra fuzz?").
    """
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / min(area_a, area_b)


def iou_over_union(a: Box, b: Box) -> float:
    """Intersection as a fraction of the two boxes' COMBINED area - the
    textbook IoU. Unlike iou_over_smaller, a big box containing a much
    smaller one scores *low* here (the small box barely dents the union),
    so this only fires for two boxes that are close to the same size and
    place - i.e. actual duplicates, not "one box happens to contain
    another, unrelated, smaller thing."
    """
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


def dedupe(boxes: list[Box], thresh: float = 0.4) -> list[Box]:
    """Merge near-duplicate boxes (e.g. a border traced as two close-fitting
    nested contours). Uses IoU-over-union so a genuinely distinct box isn't
    dropped just because it happens to sit inside a larger spurious one."""
    boxes = sorted(boxes, key=lambda b: -(b[2] - b[0]) * (b[3] - b[1]))
    kept: list[Box] = []
    for b in boxes:
        if any(iou_over_union(b, k) > thresh for k in kept):
            continue
        kept.append(b)
    return kept


def find_bordered_boxes(
    edges: np.ndarray,
    page_area: int,
    w: int,
    h: int,
    fill_thresh: float,
    min_area_frac: float,
    min_side_frac: float,
) -> list[Box]:
    """Stage A: find artworks via their clean rectangular border.

    See the "HOW THIS WORKS" note at the top of the file for the concepts
    (Canny edges, morphology, contours, fill ratio) this leans on.
    """
    # A tiny CLOSE seals any 1-2px gaps Canny left in an otherwise-solid
    # border line, so the whole rectangle traces as one connected blob
    # instead of several disconnected dashes.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    # RETR_LIST: give us every blob, not just the outermost one (see the
    # RETR_EXTERNAL gotcha in the module docstring) - this is why we then
    # have to filter out things that aren't actually artwork ourselves.
    contours, _ = cv2.findContours(closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for c in contours:
        x, y, bw, bh = cv2.boundingRect(c)  # smallest upright rect around the blob
        area = bw * bh
        if area < min_area_frac * page_area:
            continue  # too small to be a reproduction (dust, a stray mark)
        if bw < min_side_frac * w or bh < min_side_frac * h:
            continue  # too thin in one direction to be a photo (a text line, a rule)
        if bw > 0.85 * w or bh > 0.85 * h:
            continue  # the outer page/scan frame, or a gutter/spine strip
        # The fill-ratio test: does this blob's traced outline actually
        # enclose a filled rectangle, or just a sparse/ragged shape (like a
        # block of caption text, which "boundingRect"s into a rectangle but
        # is mostly whitespace inside)? See the primer for why this works.
        fill_ratio = cv2.contourArea(c) / area if area else 0
        if fill_ratio < fill_thresh:
            continue
        boxes.append((x, y, x + bw, y + bh))
    return dedupe(boxes)


def trim_box(edges: np.ndarray, box: Box, thresh: float, min_run: int) -> Box:
    """Shrink a box inward on each side while the edge density stays low,
    to cut off a low-density tail (typically adjacent caption text).

    `region` is a 2D True/False array (edge pixel or not) for the box.
    Averaging a boolean array gives the fraction that's True, so
    `row_density[i]` is "what fraction of row i's pixels are edges" - the
    same density idea as find_bleeding_boxes, just computed per row/column
    instead of for the whole box at once. We then walk in from each side
    `min_run` pixels at a time, stopping as soon as we hit a run of rows (or
    columns) dense enough to plausibly still be inside the painting.
    """
    x0, y0, x1, y1 = box
    region = edges[y0:y1, x0:x1] > 0
    row_density = region.mean(axis=1)
    col_density = region.mean(axis=0)

    top = 0
    while top < len(row_density) - min_run and row_density[top : top + min_run].mean() < thresh:
        top += 1
    bottom = len(row_density)
    while bottom > top + min_run and row_density[bottom - min_run : bottom].mean() < thresh:
        bottom -= 1
    left = 0
    while left < len(col_density) - min_run and col_density[left : left + min_run].mean() < thresh:
        left += 1
    right = len(col_density)
    while right > left + min_run and col_density[right - min_run : right].mean() < thresh:
        right -= 1

    return (x0 + left, y0 + top, x0 + right, y0 + bottom)


def find_bleeding_boxes(
    edges: np.ndarray,
    bordered: list[Box],
    page_area: int,
    w: int,
    h: int,
    border_strip: int,
    dilate_ksize: int,
    min_area_frac: float,
    min_density: float,
    trim_thresh: float,
    trim_min_run: int,
) -> list[Box]:
    """Stage B: find artworks with no border, by looking for a big enough
    patch of densely-packed edges instead (see the primer at the top of the
    file). Runs only on leftover space Stage A didn't already claim.
    """
    # The scanned spread has a faint rectangular frame around its own outer
    # edge. A bleeding artwork's own boundary touches that frame directly
    # (there's no white margin between them), so without this step the
    # frame acts as a wire connecting the artwork to *everything else* that
    # also happens to touch the frame - findContours would then trace the
    # whole mess as one blob covering nearly the entire page. Blanking a
    # strip at the very edge severs that connection before we ever look for
    # contours, at the minor cost of losing a sliver of the artwork's own
    # edge pixels there.
    stripped = edges.copy()
    m = border_strip
    stripped[:m, :] = 0
    stripped[-m:, :] = 0
    stripped[:, :m] = 0
    stripped[:, -m:] = 0

    # A much bigger dilate than Stage A's: we're not sealing hairline gaps
    # in a border now, we're deliberately fusing an artwork's internal
    # texture (brushstrokes, tonal edges) into one solid blob. Bigger
    # kernel = bridges bigger gaps = more forgiving of sparse/flat regions
    # (like a hazy sky) but more likely to also fuse in nearby unrelated
    # content, so this is the main knob to retune per scan set.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (dilate_ksize, dilate_ksize))
    dilated = cv2.dilate(stripped, kernel, iterations=1)
    contours, _ = cv2.findContours(dilated, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for c in contours:
        x, y, bw, bh = cv2.boundingRect(c)
        area = bw * bh
        if area < min_area_frac * page_area:
            continue  # too small to be a reproduction
        if bw > 0.97 * w and bh > 0.97 * h:
            continue  # the whole page merged into one blob despite the stripping
        # Density is measured on the *un-dilated* `stripped` mask, not the
        # fused `dilated` one - dilate was only a tool to decide the box's
        # extent. If we measured density post-dilate it would read high
        # for everything (that's the whole point of dilating), telling us
        # nothing about whether the content underneath was actually dense.
        region = stripped[y : y + bh, x : x + bw]
        density = cv2.countNonZero(region) / area
        if density < min_density:
            continue
        box = (x, y, x + bw, y + bh)
        if any(iou_over_smaller(box, a) > 0.3 for a in bordered):
            continue  # already found via the border pass
        box = trim_box(edges, box, trim_thresh, trim_min_run)
        boxes.append(box)
    return dedupe(boxes)


def detect_boxes(
    img: np.ndarray,
    canny_low: int = 40,
    canny_high: int = 120,
    fill_thresh: float = 0.85,
    min_area_frac: float = 0.008,
    min_side_frac: float = 0.04,
    border_strip: int = 15,
    bleed_dilate: int = 11,
    bleed_min_area_frac: float = 0.02,
    bleed_min_density: float = 0.10,
    trim_thresh: float = 0.06,
    trim_min_run: int = 15,
) -> list[Box]:
    h, w = img.shape[:2]
    page_area = h * w

    # Common preprocessing for both stages - see the primer at the top of
    # the file for what each step does and why.
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(gray, canny_low, canny_high)

    bordered = find_bordered_boxes(
        edges, page_area, w, h, fill_thresh, min_area_frac, min_side_frac
    )
    bleeding = find_bleeding_boxes(
        edges,
        bordered,
        page_area,
        w,
        h,
        border_strip,
        bleed_dilate,
        bleed_min_area_frac,
        bleed_min_density,
        trim_thresh,
        trim_min_run,
    )
    # A box that sits almost entirely inside another is virtually always the
    # same artwork picked up twice (e.g. a dense sub-region of a bleeding
    # painting also clearing the blob thresholds on its own).
    boxes = sorted(bordered + bleeding, key=lambda b: -(b[2] - b[0]) * (b[3] - b[1]))
    kept: list[Box] = []
    for b in boxes:
        if any(iou_over_smaller(b, k) > 0.75 for k in kept):
            continue
        kept.append(b)

    kept.sort(key=lambda b: (b[1], b[0]))  # reading order: top-to-bottom, left-to-right
    return kept


def draw_debug(img: np.ndarray, boxes: list[Box]) -> np.ndarray:
    out = img.copy()
    for i, (x0, y0, x1, y1) in enumerate(boxes, start=1):
        cv2.rectangle(out, (x0, y0), (x1, y1), (0, 0, 255), 2)
        cv2.putText(
            out, str(i), (x0 + 4, y0 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pages-dir", type=Path, default=PAGES_DIR)
    ap.add_argument("--out-json", type=Path, default=OUT_JSON)
    ap.add_argument("--debug-dir", type=Path, default=DEBUG_DIR)
    ap.add_argument("--no-debug", action="store_true", help="skip writing annotated PNGs")
    ap.add_argument("--canny-low", type=int, default=40)
    ap.add_argument("--canny-high", type=int, default=120)
    ap.add_argument("--fill-thresh", type=float, default=0.85, help="stage A: min polygon/bbox fill ratio")
    ap.add_argument("--min-area-frac", type=float, default=0.008, help="stage A: min box area, as a fraction of the page")
    ap.add_argument("--min-side-frac", type=float, default=0.04, help="stage A: min box width/height, as a fraction of the page")
    ap.add_argument("--border-strip", type=int, default=15, help="stage B: px blanked at the page's outer edge")
    ap.add_argument("--bleed-dilate", type=int, default=11, help="stage B: dilation kernel size (px)")
    ap.add_argument("--bleed-min-area-frac", type=float, default=0.02, help="stage B: min box area, as a fraction of the page")
    ap.add_argument("--bleed-min-density", type=float, default=0.10, help="stage B: min edge-pixel density within the box")
    args = ap.parse_args()

    pages = sorted(args.pages_dir.glob("*_small.png"))
    if not pages:
        raise SystemExit(f"No *_small.png files found in {args.pages_dir}/")

    if not args.no_debug:
        args.debug_dir.mkdir(parents=True, exist_ok=True)

    results = []
    total = 0
    for page_path in pages:
        img = cv2.imread(str(page_path))
        h, w = img.shape[:2]
        boxes = detect_boxes(
            img,
            canny_low=args.canny_low,
            canny_high=args.canny_high,
            fill_thresh=args.fill_thresh,
            min_area_frac=args.min_area_frac,
            min_side_frac=args.min_side_frac,
            border_strip=args.border_strip,
            bleed_dilate=args.bleed_dilate,
            bleed_min_area_frac=args.bleed_min_area_frac,
            bleed_min_density=args.bleed_min_density,
        )
        total += len(boxes)

        results.append(
            {
                "image": page_path.name,
                "image_size": [w, h],
                "items": [{"bbox": list(b)} for b in boxes],
            }
        )
        print(f"{page_path.name}: {len(boxes)} boxes")

        if not args.no_debug:
            debug_img = draw_debug(img, boxes)
            cv2.imwrite(str(args.debug_dir / page_path.name), debug_img)

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n{total} boxes total. Wrote {args.out_json}")
    if not args.no_debug:
        print(f"Wrote annotated previews to {args.debug_dir}/")


if __name__ == "__main__":
    main()
