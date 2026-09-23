#!/usr/bin/env python3
"""Losslessly rasterize each page of every PDF in pdfs/ to PNG images in pages/.

Output naming: <PDFSTEM>_<zero-padded page number>.png, e.g. KIC_000.png
"""

import sys
from pathlib import Path

import pymupdf

PDFS_DIR = Path("pdfs")
PAGES_DIR = Path("pages")
DPI = 300  # rasterization resolution (PNG itself is lossless)
PAD = 3  # page number padding, e.g. 000


def extract(pdf_path: Path) -> None:
    prefix = pdf_path.stem.upper()
    doc = pymupdf.open(pdf_path)
    zoom = DPI / 72  # PDF native resolution is 72 dpi
    try:
        for i, page in enumerate(doc):
            pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), alpha=False)
            out = PAGES_DIR / f"{prefix}_{i:0{PAD}d}.png"
            pix.save(out)  # PNG = lossless
            print(f"{out} ({pix.width}x{pix.height})")
    finally:
        doc.close()


def main() -> None:
    pdfs = sorted(PDFS_DIR.glob("*.pdf"))
    if not pdfs:
        sys.exit(f"No PDFs found in {PDFS_DIR}/")
    PAGES_DIR.mkdir(exist_ok=True)
    for pdf in pdfs:
        print(f"Processing {pdf} ...")
        extract(pdf)


if __name__ == "__main__":
    main()
