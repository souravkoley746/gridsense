"""Capture live screenshots of all 6 GridSense Streamlit dashboard pages operating on genuine UCI data."""
import asyncio
from pathlib import Path
from playwright.async_api import async_playwright

ROOT = Path(__file__).resolve().parents[1]
SCREENSHOTS_DIR = ROOT / "screenshots"
ARTIFACT_DIR = Path(r"C:\Users\sourav\.gemini\antigravity\brain\a2f65c3b-8d2c-45a4-b7b5-6900fae8d669")
PAGES = [
    ("Overview", "page_1_overview.png"),
    ("Historical Analysis", "page_2_historical.png"),
    ("Forecasting", "page_3_forecasting.png"),
    ("Model Performance", "page_4_performance.png"),
    ("Data Quality", "page_5_quality.png"),
    ("About Project", "page_6_about.png"),
]

async def capture():
    SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(viewport={"width": 1440, "height": 1080})
        page = await context.new_page()
        
        print("Navigating to http://127.0.0.1:8501...")
        await page.goto("http://127.0.0.1:8501", wait_until="networkidle", timeout=60000)
        await page.wait_for_timeout(3000)
        
        # Take Page 1 Overview
        out_root_1 = SCREENSHOTS_DIR / PAGES[0][1]
        out_art_1 = ARTIFACT_DIR / PAGES[0][1]
        out_real_art_1 = ARTIFACT_DIR / f"real_{PAGES[0][1]}"
        await page.screenshot(path=str(out_root_1), full_page=True)
        await page.screenshot(path=str(out_art_1), full_page=True)
        await page.screenshot(path=str(out_real_art_1), full_page=True)
        print(f"Captured {PAGES[0][0]} -> {out_root_1}")
        
        # Now click through pages
        for page_name, filename in PAGES[1:]:
            print(f"Navigating to {page_name}...")
            target = page.get_by_text(page_name, exact=False).first
            await target.click()
            await page.wait_for_timeout(3000)
            
            # If forecasting page, trigger forecast button to capture fan chart
            if page_name == "Forecasting":
                btn = page.get_by_role("button", name="GENERATE FORECAST").first
                if await btn.count() > 0:
                    await btn.click()
                    await page.wait_for_timeout(3000)
                    
            out_root = SCREENSHOTS_DIR / filename
            out_art = ARTIFACT_DIR / filename
            out_real_art = ARTIFACT_DIR / f"real_{filename}"
            await page.screenshot(path=str(out_root), full_page=True)
            await page.screenshot(path=str(out_art), full_page=True)
            await page.screenshot(path=str(out_real_art), full_page=True)
            print(f"Captured {page_name} -> {out_root}")
            
        await browser.close()
        print("Screenshot capture complete!")

if __name__ == "__main__":
    asyncio.run(capture())
