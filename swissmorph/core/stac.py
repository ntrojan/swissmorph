"""StacDownloader - fetches and mosaics swissALTI3D tiles.

Downloads DTM tiles from the swisstopo STAC API and mosaics them
into a single GeoTIFF in the target CRS (EPSG:2056 by default).

Pure Python / rasterio / numpy - NO QGIS imports.
"""

import json
import os
import re
import shutil
import time
import urllib.error
import urllib.request

# Parses the grid-position token from swissALTI3D item IDs:
#   swissalti3d_2019_2730-1118_2_2056_5728  →  pos="2730-1118"
_TILE_ID_RE = re.compile(r"^swissalti3d_(\d{4})_([\d-]+)_")

import rasterio
from rasterio.merge import merge


class StacDownloader:
    """Download and mosaic swissALTI3D tiles from the swisstopo STAC API."""

    _UA = "SwissMorph-QGIS-Plugin/0.1.0"

    def __init__(self, tmp_dir: str, config: dict | None = None):
        self._tmp_dir = tmp_dir
        self._cfg = config if config is not None else self._load_config()

    def _load_config(self) -> dict:
        """Load config/defaults.json relative to this file (core/stac.py)."""
        cfg_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..", "config", "defaults.json",
        )
        with open(cfg_path, encoding="utf-8") as f:
            return json.load(f)

    def fetch(self, bbox_lv95: tuple, progress_callback=None) -> str:
        """Download tiles for *bbox_lv95* and return path to the mosaicked GeoTIFF.

        Args:
            bbox_lv95:         (xmin, ymin, xmax, ymax) in EPSG:2056.
            progress_callback: callable(str) for progress messages.

        Returns:
            Absolute path to the mosaicked GeoTIFF inside tmp_dir.

        Raises:
            RuntimeError: if no tiles are found or the download fails.
        """
        cb = progress_callback or (lambda _: None)

        # 1. Reproject AOI to WGS84 for the STAC bbox query
        bbox_wgs84 = self._to_wgs84(bbox_lv95)
        w, s, e, n = bbox_wgs84
        cb(f"STAC query bbox (WGS84): W={w:.4f} S={s:.4f} E={e:.4f} N={n:.4f}")

        # 2. Query STAC items, then keep only the most recent vintage per grid cell
        items = self._query_items(bbox_wgs84, cb)
        if not items:
            raise RuntimeError(
                f"No swissALTI3D tiles found for bbox {bbox_lv95}. "
                "Check that the AOI is within Switzerland."
            )
        items = self._deduplicate_items(items, cb)
        cb(f"{len(items)} tile(s) to download.")

        # 3. Download each tile (with streaming + retry)
        tile_paths = []
        for i, item in enumerate(items, 1):
            href = self._select_asset_href(item)
            if href is None:
                cb(f"  Warning: No suitable asset in item '{item.get('id', '')}' - skipped.")
                continue
            tile_path = os.path.join(self._tmp_dir, f"tile_{i:04d}.tif")
            cb(f"  Downloading tile {i}/{len(items)}: {os.path.basename(href)}")
            self._download_tile(href, tile_path, cb, i, len(items))
            tile_paths.append(tile_path)

        if not tile_paths:
            raise RuntimeError("All tiles skipped (no suitable GeoTIFF assets found).")

        # 4. Mosaic
        mosaic_path = os.path.join(self._tmp_dir, "dtm_mosaic.tif")
        if len(tile_paths) == 1:
            cb("  Single tile - skipping mosaic step.")
            os.rename(tile_paths[0], mosaic_path)
        else:
            cb(f"  Mosaicking {len(tile_paths)} tiles...")
            datasets = [rasterio.open(p) for p in tile_paths]
            try:
                mosaic, transform = merge(datasets)
                profile = datasets[0].profile.copy()
                profile.update({
                    "driver":    "GTiff",
                    "height":    mosaic.shape[1],
                    "width":     mosaic.shape[2],
                    "transform": transform,
                    "compress":  "deflate",
                    "tiled":     True,
                    "count":     1,
                })
                with rasterio.open(mosaic_path, "w", **profile) as dst:
                    dst.write(mosaic)
            finally:
                for ds in datasets:
                    ds.close()
            cb(f"  Mosaic written: {mosaic_path}")

        return mosaic_path

    def _download_tile(
        self,
        href: str,
        dest: str,
        cb,
        tile_num: int,
        total: int,
        max_retries: int = 3,
    ) -> None:
        """Download *href* to *dest*, retrying up to *max_retries* times.

        Uses streaming (copyfileobj) to avoid loading the entire tile into RAM.
        Exponential backoff: 1 s, 2 s, 4 s between attempts.
        """
        for attempt in range(1, max_retries + 1):
            try:
                req = urllib.request.Request(href, headers={"User-Agent": self._UA})
                with urllib.request.urlopen(req, timeout=120) as resp:
                    with open(dest, "wb") as fh:
                        shutil.copyfileobj(resp, fh)
                return
            except Exception as exc:
                if attempt < max_retries:
                    wait = 2 ** (attempt - 1)
                    cb(
                        f"    Tile {tile_num}/{total}: error ({exc}), "
                        f"retry {attempt}/{max_retries - 1} in {wait} s…"
                    )
                    time.sleep(wait)
                else:
                    raise RuntimeError(
                        f"Tile {tile_num}/{total} download failed after "
                        f"{max_retries} attempts: {exc}"
                    ) from exc

    def _to_wgs84(self, bbox_lv95: tuple) -> tuple:
        """Reproject a bounding box from EPSG:2056 to EPSG:4326.

        Args:
            bbox_lv95: (xmin, ymin, xmax, ymax) in EPSG:2056 (LV95).

        Returns:
            (lon_min, lat_min, lon_max, lat_max) in WGS84.
        """
        from pyproj import Transformer
        xmin, ymin, xmax, ymax = bbox_lv95
        tr = Transformer.from_crs("EPSG:2056", "EPSG:4326", always_xy=True)
        lon_min, lat_min = tr.transform(xmin, ymin)
        lon_max, lat_max = tr.transform(xmax, ymax)
        return lon_min, lat_min, lon_max, lat_max

    def _query_items(self, bbox_wgs84: tuple, cb) -> list:
        """Query STAC items intersecting *bbox_wgs84*, following pagination.

        Endpoint: GET {base_url}/collections/{collection}/items
        Parameters: bbox={w},{s},{e},{n}&limit={limit}

        Args:
            bbox_wgs84: (lon_min, lat_min, lon_max, lat_max) in WGS84.
            cb:         progress callback(str).

        Returns:
            list[dict]: All STAC item feature dicts across pages.
        """
        base = self._cfg["stac"]["base_url"].rstrip("/")
        coll = self._cfg["stac"]["collection"]
        lim  = self._cfg["stac"].get("items_limit", 100)
        w, s, e, n = bbox_wgs84

        url = (
            f"{base}/collections/{coll}/items"
            f"?bbox={w},{s},{e},{n}&limit={lim}"
        )
        items = []
        page  = 1

        while url:
            cb(f"Fetching page {page}...")
            data = self._get_json(url)
            items.extend(data.get("features", []))

            # Follow the "next" pagination link
            url = next(
                (lnk["href"] for lnk in data.get("links", []) if lnk.get("rel") == "next"),
                None,
            )
            page += 1

        return items

    def _get_json(self, url: str) -> dict:
        """HTTP GET *url* and return parsed JSON.

        Args:
            url: Full URL to request.

        Returns:
            Parsed JSON response as dict.

        Raises:
            RuntimeError: on HTTP error responses.
        """
        req = urllib.request.Request(url, headers={"User-Agent": self._UA})
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f"STAC HTTP {exc.code} for {url}") from exc

    def _deduplicate_items(self, items: list, cb) -> list:
        """Keep only the most recent STAC item per spatial tile position.

        The swisstopo STAC returns tiles for every acquisition year that
        intersects the bbox (e.g. 2019, 2020, 2022, 2023).  For each spatial
        grid cell only the item with the latest ``properties.datetime`` is kept.

        Position key: rounded WGS84 bbox SW corner (a STAC-required field,
        identical across years for the same physical tile location).
        """
        by_pos: dict[str, tuple[str, dict]] = {}

        for item in items:
            bbox = item.get("bbox")
            if bbox and len(bbox) >= 2:
                pos_key = f"{bbox[0]:.4f},{bbox[1]:.4f}"
            else:
                pos_key = item.get("id", str(id(item)))

            dt = (item.get("properties") or {}).get("datetime") or ""
            if pos_key not in by_pos or dt > by_pos[pos_key][0]:
                by_pos[pos_key] = (dt, item)

        result = [v[1] for v in by_pos.values()]
        n_dup = len(items) - len(result)
        if n_dup:
            cb(
                f"  {n_dup} older-vintage tile(s) removed — "
                f"{len(result)} unique position(s) will be downloaded."
            )
        return result

    def _select_asset_href(self, item: dict) -> str | None:
        """Select the best GeoTIFF asset from a STAC item.

        Selection priority (sort key, lower is better):
          1. EPSG match: prefer proj:epsg == target_epsg (0), mismatch (1), unknown (2).
          2. Resolution delta: prefer |eo:gsd - target_resolution_m| closest to 0.

        Falls back to any image/tiff asset if no proj:epsg metadata is present.

        Args:
            item: STAC item dict with an "assets" key.

        Returns:
            href string of the best asset, or None if no GeoTIFF found.
        """
        target_epsg = self._cfg["stac"].get("target_epsg", 2056)
        target_res  = self._cfg["stac"].get("target_resolution_m", 2.0)

        candidates = []
        for _key, asset in item.get("assets", {}).items():
            mime = asset.get("type", "")
            if "image/tiff" not in mime and "geotiff" not in mime.lower():
                continue
            epsg      = asset.get("proj:epsg")
            gsd       = asset.get("eo:gsd")
            res_delta = abs(gsd - target_res) if gsd is not None else float("inf")

            if epsg is None:
                epsg_score = 2   # no metadata - lowest priority
            elif epsg == target_epsg:
                epsg_score = 0   # exact match
            else:
                epsg_score = 1   # wrong epsg

            candidates.append((epsg_score, res_delta, asset["href"]))

        if not candidates:
            return None

        # Sort key: (epsg_mismatch, res_delta) - lower is better
        candidates.sort(key=lambda c: (c[0], c[1]))
        return candidates[0][2]
