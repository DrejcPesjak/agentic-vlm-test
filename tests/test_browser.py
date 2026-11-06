"""Opt-in browser test: VLM_BROWSER_TESTS=1 pytest tests/test_browser.py."""

from __future__ import annotations

import os
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

import pytest
from playwright.sync_api import sync_playwright
from test_experiment import FakeCaller, make_dataset

from vlm_crop.budget import Ledger
from vlm_crop.core import Config
from vlm_crop.report import build_report
from vlm_crop.runner import execute

pytestmark = pytest.mark.skipif(
    os.getenv("VLM_BROWSER_TESTS") != "1", reason="Set VLM_BROWSER_TESTS=1 with Chromium installed"
)


def test_static_subpath_replay_desktop_mobile(tmp_path):
    data = tmp_path / "data"
    make_dataset(data)
    run = tmp_path / "run"
    execute(
        data, run, Config(("gpt-4o",)), FakeCaller(exhaust=True), Ledger(tmp_path / "budget.json")
    )
    build_report(run, tmp_path / "project-subpath")
    server = ThreadingHTTPServer(
        ("127.0.0.1", 0), partial(SimpleHTTPRequestHandler, directory=str(tmp_path))
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch()
            page = browser.new_page(viewport={"width": 1440, "height": 1100})
            errors = []
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.goto(f"http://127.0.0.1:{server.server_port}/project-subpath/")
            assert page.locator(".panel").count() == 3
            assert page.locator("#scope").inner_text().startswith("Pilot smoke test")
            page.locator("#round").fill("2")
            assert page.locator(".crop-box").count() == 1
            assert (
                page.locator(".panel").first.locator(".subtitle").inner_text().endswith("finished")
            )
            page.locator("#play").click()
            page.wait_for_function("document.getElementById('round').value === '5'")
            assert page.locator("#play").inner_text() == "Play replay"
            assert page.locator("#png").get_attribute("href").startswith("exports/")
            assert page.locator("#gif").get_attribute("href").endswith(".gif")
            page.wait_for_function(
                "[...document.images].every(i => i.complete && i.naturalWidth > 0)"
            )
            page.set_viewport_size({"width": 390, "height": 844})
            assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
            page.select_option("#outcome", "improvement")
            assert page.locator("#empty").is_visible()
            page.select_option("#outcome", "all")
            assert page.locator("#replay").is_visible()
            assert errors == []
            browser.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
