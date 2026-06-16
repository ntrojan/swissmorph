"""Unit tests for core/stac.py.

Tests that do NOT require network access (asset selection, bbox
reprojection, config loading). HTTP calls are mocked where needed.

Run without QGIS:
    python -m unittest discover -s swissmorph/tests -v
"""

import json
import math
import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from swissmorph.core.stac import StacDownloader


def _downloader(tmp_dir: str, cfg: dict | None = None) -> StacDownloader:
    cfg = cfg or {
        "stac": {
            "base_url":            "https://data.geo.admin.ch/api/stac/v0.9",
            "collection":          "ch.swisstopo.swissalti3d",
            "target_epsg":         2056,
            "target_resolution_m": 2.0,
            "items_limit":         100,
        },
        "crs":   {"default": "EPSG:2056", "stac_query": "EPSG:4326"},
        "cache": {"dir_prefix": "swissmorph_"},
    }
    return StacDownloader(tmp_dir, config=cfg)


# Asset selection

class TestSelectAssetHref(unittest.TestCase):

    def _item(self, assets: dict) -> dict:
        return {"id": "test-item", "assets": assets}

    def test_prefers_target_epsg_and_resolution(self):
        """Asset with proj:epsg=2056 and eo:gsd~2.0 must win."""
        item = self._item({
            "asset_2m_lv95": {
                "href": "https://example.com/tile_2m_2056.tif",
                "type": "image/tiff; application=geotiff",
                "proj:epsg": 2056,
                "eo:gsd": 2.0,
            },
            "asset_0.5m_lv95": {
                "href": "https://example.com/tile_0.5m_2056.tif",
                "type": "image/tiff; application=geotiff",
                "proj:epsg": 2056,
                "eo:gsd": 0.5,
            },
            "asset_wgs84": {
                "href": "https://example.com/tile_wgs84.tif",
                "type": "image/tiff; application=geotiff",
                "proj:epsg": 4326,
                "eo:gsd": 2.0,
            },
        })
        with tempfile.TemporaryDirectory() as tmp:
            href = _downloader(tmp)._select_asset_href(item)
        self.assertEqual(href, "https://example.com/tile_2m_2056.tif")

    def test_falls_back_to_any_geotiff_when_no_epsg_metadata(self):
        """If no asset has proj:epsg, fall back to any GeoTIFF."""
        item = self._item({
            "asset_a": {
                "href": "https://example.com/tile.tif",
                "type": "image/tiff",
            }
        })
        with tempfile.TemporaryDirectory() as tmp:
            href = _downloader(tmp)._select_asset_href(item)
        self.assertIsNotNone(href)
        self.assertEqual(href, "https://example.com/tile.tif")

    def test_returns_none_when_no_geotiff_asset(self):
        """Items without any image/tiff asset -> None."""
        item = self._item({
            "meta": {"href": "https://example.com/meta.json", "type": "application/json"}
        })
        with tempfile.TemporaryDirectory() as tmp:
            result = _downloader(tmp)._select_asset_href(item)
        self.assertIsNone(result)

    def test_empty_assets_returns_none(self):
        item = self._item({})
        with tempfile.TemporaryDirectory() as tmp:
            result = _downloader(tmp)._select_asset_href(item)
        self.assertIsNone(result)

    def test_closest_resolution_wins_among_same_epsg(self):
        """Among two 2056 assets, the one closest to target_res (2.0) wins."""
        item = self._item({
            "high_res": {
                "href": "https://example.com/0.5m.tif",
                "type": "image/tiff",
                "proj:epsg": 2056,
                "eo:gsd": 0.5,    # |0.5 - 2.0| = 1.5
            },
            "target_res": {
                "href": "https://example.com/2m.tif",
                "type": "image/tiff",
                "proj:epsg": 2056,
                "eo:gsd": 2.0,    # |2.0 - 2.0| = 0.0  <- wins
            },
        })
        with tempfile.TemporaryDirectory() as tmp:
            href = _downloader(tmp)._select_asset_href(item)
        self.assertEqual(href, "https://example.com/2m.tif")

    def test_custom_target_resolution(self):
        """Config with target_resolution_m=0.5 -> picks the 0.5 m asset."""
        cfg = {
            "stac": {
                "base_url": "https://x", "collection": "c",
                "target_epsg": 2056, "target_resolution_m": 0.5, "items_limit": 10
            }
        }
        item = self._item({
            "a": {"href": "https://x/0.5m.tif", "type": "image/tiff",
                  "proj:epsg": 2056, "eo:gsd": 0.5},
            "b": {"href": "https://x/2m.tif",   "type": "image/tiff",
                  "proj:epsg": 2056, "eo:gsd": 2.0},
        })
        with tempfile.TemporaryDirectory() as tmp:
            href = _downloader(tmp, cfg)._select_asset_href(item)
        self.assertEqual(href, "https://x/0.5m.tif")


# Bbox reprojection

class TestToWGS84(unittest.TestCase):

    def test_output_within_switzerland(self):
        """LV95 bbox of central Switzerland -> WGS84 bbox inside CH bounds."""
        try:
            from pyproj import Transformer
        except ImportError:
            self.skipTest("pyproj not available")

        # Approximate centre of Switzerland in LV95
        bbox_lv95 = (620000.0, 180000.0, 780000.0, 270000.0)
        with tempfile.TemporaryDirectory() as tmp:
            lon_min, lat_min, lon_max, lat_max = _downloader(tmp)._to_wgs84(bbox_lv95)

        # Switzerland bounding box in WGS84
        self.assertGreater(lon_min, 5.9,  "lon_min must be east of 5.9 deg")
        self.assertLess(lon_max,    10.6,  "lon_max must be west of 10.6 deg")
        self.assertGreater(lat_min, 45.8,  "lat_min must be north of 45.8 deg")
        self.assertLess(lat_max,    47.9,  "lat_max must be south of 47.9 deg")
        self.assertLess(lon_min, lon_max,  "lon_min < lon_max")
        self.assertLess(lat_min, lat_max,  "lat_min < lat_max")

    def test_small_bbox_stays_small(self):
        """A 1 km2 bbox in LV95 must map to < 0.02 deg in WGS84."""
        try:
            from pyproj import Transformer
        except ImportError:
            self.skipTest("pyproj not available")

        bbox_lv95 = (700000.0, 200000.0, 701000.0, 201000.0)  # 1 km2
        with tempfile.TemporaryDirectory() as tmp:
            lon_min, lat_min, lon_max, lat_max = _downloader(tmp)._to_wgs84(bbox_lv95)

        self.assertLess(lon_max - lon_min, 0.02)
        self.assertLess(lat_max - lat_min, 0.02)


# Config loading

class TestLoadConfig(unittest.TestCase):

    def test_loads_defaults_json(self):
        """_load_config reads config/defaults.json relative to core/stac.py."""
        with tempfile.TemporaryDirectory() as tmp:
            d = StacDownloader(tmp)   # triggers _load_config via __init__
        self.assertIn("stac", d._cfg)
        self.assertIn("base_url", d._cfg["stac"])
        self.assertIn("target_epsg", d._cfg["stac"])

    def test_custom_config_overrides_defaults(self):
        """Passing config= must bypass _load_config entirely."""
        custom = {"stac": {"base_url": "https://custom", "collection": "col",
                            "target_epsg": 9999, "target_resolution_m": 5.0,
                            "items_limit": 1}}
        with tempfile.TemporaryDirectory() as tmp:
            d = _downloader(tmp, cfg=custom)
        self.assertEqual(d._cfg["stac"]["target_epsg"], 9999)


# Pagination

class TestQueryItemsPagination(unittest.TestCase):

    def _page(self, n_items: int, next_url: str | None = None) -> dict:
        features = [{"id": f"item-{i}", "assets": {}} for i in range(n_items)]
        links = [{"rel": "next", "href": next_url}] if next_url else []
        return {"features": features, "links": links}

    def test_follows_next_link(self):
        """_query_items must follow pagination and accumulate all items."""
        page1 = self._page(3, next_url="https://stac/page2")
        page2 = self._page(2, next_url=None)

        with tempfile.TemporaryDirectory() as tmp:
            d = _downloader(tmp)
            with patch.object(d, "_get_json", side_effect=[page1, page2]):
                items = d._query_items((6.0, 46.0, 8.0, 47.0), lambda _: None)

        self.assertEqual(len(items), 5)

    def test_single_page_no_next(self):
        """Single-page response (no next link) returns the correct items."""
        page = self._page(4, next_url=None)

        with tempfile.TemporaryDirectory() as tmp:
            d = _downloader(tmp)
            with patch.object(d, "_get_json", return_value=page):
                items = d._query_items((6.0, 46.0, 8.0, 47.0), lambda _: None)

        self.assertEqual(len(items), 4)

    def test_empty_response_returns_empty_list(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = _downloader(tmp)
            with patch.object(d, "_get_json", return_value={"features": [], "links": []}):
                items = d._query_items((6.0, 46.0, 8.0, 47.0), lambda _: None)
        self.assertEqual(items, [])


if __name__ == "__main__":
    unittest.main()
