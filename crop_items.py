import json
import os

from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
JSON_PATH = os.path.join(BASE_DIR, "image-captions", "KIC.json")
PAGES_DIR = os.path.join(BASE_DIR, "pages")
OUT_DIR = os.path.join(BASE_DIR, "image-captions")

with open(JSON_PATH, encoding="utf-8") as f:
    text = f.read()

# The file may contain a prose preamble before the JSON array.
start = text.find("[")
end = text.rfind("]")
data = json.loads(text[start : end + 1])

item_number = 1
for page in data:  # each entry is a page/spread object
    src = os.path.join(PAGES_DIR, page["image"])
    im = Image.open(src)
    for item in page["items"]:
        xmin, ymin, xmax, ymax = item["bbox"]  # [xmin, ymin, xmax, ymax] in pixels
        crop_box = (xmin, ymin, xmax, ymax)
        out_name = f"{page['image']}_{item_number:03d}.png"
        out_path = os.path.join(OUT_DIR, out_name)
        im.crop(crop_box).save(out_path)
        item_number += 1

print(f"Exported {item_number - 1} crops to {OUT_DIR}")
