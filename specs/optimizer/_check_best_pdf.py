"""Rasterize the best-model PDF pages to PNGs for visual verification."""
import fitz

PDF = r"D:\Personal\context-health-factory\specs\optimizer\factory_best_model.pdf"
doc = fitz.open(PDF)
print("pages:", len(doc))
for i, page in enumerate(doc):
    pix = page.get_pixmap(dpi=100)
    out = rf"D:\Personal\context-health-factory\specs\optimizer\_best_p{i+1}.png"
    pix.save(out)
    print(i + 1, page.rect, "->", out)
