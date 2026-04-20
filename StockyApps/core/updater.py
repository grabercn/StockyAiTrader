"""
Update Checker — checks GitHub releases for newer versions.

Runs once on startup in a background thread. If a newer version exists,
notifies via the event bus (status bar + logs). Non-blocking, non-intrusive.
"""

import requests
from packaging import version as pkg_version
from .branding import APP_VERSION

REPO = "grabercn/StockyAiTrader"
RELEASES_URL = f"https://api.github.com/repos/{REPO}/releases/latest"
RELEASES_PAGE = f"https://github.com/{REPO}/releases"


def check_for_update():
    """
    Check GitHub for a newer release.

    Returns:
        (has_update: bool, latest_version: str, download_url: str, release_url: str)
        or (False, "", "", "") on failure
    """
    try:
        r = requests.get(RELEASES_URL, headers={"User-Agent": "StockyAiTrader"}, timeout=10)
        if r.status_code != 200:
            return False, "", "", ""

        data = r.json()
        latest_tag = data.get("tag_name", "").lstrip("v")
        current = APP_VERSION

        if not latest_tag:
            return False, "", "", ""

        # Compare versions
        if pkg_version.parse(latest_tag) > pkg_version.parse(current):
            # Find the Windows zip asset
            download_url = ""
            for asset in data.get("assets", []):
                if "Windows" in asset.get("name", "") or asset.get("name", "").endswith(".zip"):
                    download_url = asset.get("browser_download_url", "")
                    break

            release_url = data.get("html_url", RELEASES_PAGE)
            return True, latest_tag, download_url, release_url

        return False, latest_tag, "", ""

    except Exception:
        return False, "", "", ""
