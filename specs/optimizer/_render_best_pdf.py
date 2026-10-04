"""Render factory_best_model.html to PDF (A4, backgrounds on) via headless Chromium."""
from playwright.sync_api import sync_playwright

SRC = r"D:\Personal\context-health-factory\specs\optimizer\factory_best_model.html"
OUT = r"D:\Personal\context-health-factory\specs\optimizer\factory_best_model.pdf"

with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page()
    page.goto("file:///" + SRC.replace("\\", "/"))
    page.wait_for_timeout(600)
    page.pdf(path=OUT, prefer_css_page_size=True, print_background=True)
    browser.close()
print("PDF written:", OUT)
