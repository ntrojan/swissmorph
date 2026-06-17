"""StacDownloader — fetches and mosaics swissALTI3D tiles.

Pure Python / rasterio / pyproj. No QGIS imports.
The caller (MorphometryAlgorithm) must extract (xmin, ymin, xmax, ymax)
from the QgsRectangle in EPSG:2056 before calling fetch(); this keeps
core/ fully QGIS-agnostic.

API reference:  https://data.geo.admin.ch/api/stac/v0.9/
Collection:     ch.swisstopo.swissalti3d
Tile size:      1 km²
Bbox parameter: EPSG:4326 (WGS 84), [lon_min, lat_min, lon_max, lat_max]
Asset fields:   href, type, eo:gsd (metres), proj:epsg
Pagination:     links[rel="next"] in the FeatureCollection response
"""

import json
import os
import urllib.error
import urllib.request
from typing import Callable, Optional

_UA = "SwissMorph-QGIS-Plugin/0.1.0"


class StacDownloader:
    """Query swisstopo STAC for swissALTI3D and mosaic tiles into one GeoTIFF."""

    def __init__(self, work_dir: str, config: Optional[dict] = None) -> None:
        """
        Args:
            work_dir: Directory for downloaded tiles and the output mosaic.
            config:   Optional dict overriding values from config/defaults.json.
        """
        self._work_dir = work_dir
        self._cfg = config or self._load_config()

    # ── Public API ────────────────────────────────────────────────────────────

    def fetch(
        self,
        bbox_lv95: tuple,
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> str:
        """Download tiles intersecting *bbox_lv95* and return mosaicked GeoTIFF.

        Args:
            bbox_lv95:         (xmin, ymin, xmax, ymax) in EPSG:2056 (metres).
            progress_callback: Optional callable(str) for progress messages.

        Returns:
            str: Absolute path to the mosaicked GeoTIFF in EPSG:2056.

        Raises:
            RuntimeError: If no tiles are found or any download fails.
        """
        cb = progress_callback or (lambda _: None)

        # ── 1. Reproject bbox to WGS 84 for the STAC query ───────────────────
        bbox_wgs84 = self._to_wgs84(bbox_lv95)
        cb(
            f"STAC query bbox (WGS84): "
            f"W={bbox_wgs84[0]:.4f} S={bbox_wgs84[1]:.4f} "
            f"E={bbox_wgs84[2]:.4f} N={bbox_wgs84[3]:.4f}"
        )

        # ── 2. Query items ────────────────────────────────────────────────────
        items = self._query_items(bbox_wgs84, cb)
        if not items:
            raise RuntimeError(
                "No swissALTI3D tiles found for the given area of interest. "
                "Verify that the AOI overlaps with Switzerland."
            )
        cb(f"{len(items)} item(s) returned by STAC.")

        # swissALTI3D publishes several acquisition years for the same 1 km²
        # cell. Keep only the most recent per cell so the mosaic is not a mix
        # of epochs (and tiles are not downloaded twice).
        items = self._deduplicate_latest(items, cb)
        cb(f"{len(items)} tile(s) after keeping the latest year per cell.")

        # ── 3. Resolve download URLs ──────────────────────────────────────────
        hrefs = []
        for item in items:
            href = self._select_asset_href(item)
            if href:
                hrefs.append(href)
            else:
                cb(f"  ⚠ No suitable asset in item '{item.get('id','?')}' — skipped.")

        if not hrefs:
            raise RuntimeError(
                "Items were found but none had a GeoTIFF asset in EPSG:2056. "
                "Check config.stac.target_epsg and target_resolution_m."
            )

        # ── 4. Download tiles ─────────────────────────────────────────────────
        tile_paths = self._download_tiles(hrefs, cb)

        # ── 5. Mosaic ─────────────────────────────────────────────────────────
        mosaic_path = self._mosaic(tile_paths, cb)

        # ── 6. Normalise CRS ──────────────────────────────────────────────────
        # swissALTI3D COG tiles tag their CRS as a non-standard LOCAL_CS WKT,
        # so rasterio/GDAL report crs.to_epsg() == None and QGIS loads them as
        # an "unknown" CRS that does not align with other LV95 layers. The STAC
        # asset metadata guarantees proj:epsg == target_epsg, so re-tag the
        # final DTM with a clean EPSG code.
        self._ensure_crs(mosaic_path, cb)

        cb(f"Mosaic ready: {mosaic_path}")
        return mosaic_path

    # ── Private: configuration ────────────────────────────────────────────────

    def _load_config(self) -> dict:
        """Load defaults from config/defaults.json relative to this file.

        Returns:
            dict: Configuration dictionary.
        """
        config_path = os.path.normpath(
            os.path.join(os.path.dirname(__file__), "..", "config", "defaults.json")
        )
        with open(config_path, encoding="utf-8") as fh:
            return json.load(fh)

    # ── Private: STAC query ───────────────────────────────────────────────────

    def _to_wgs84(self, bbox_lv95: tuple) -> tuple:
        """Reproject EPSG:2056 bbox to EPSG:4326 (lon/lat) for the STAC API.

        Args:
            bbox_lv95: (xmin, ymin, xmax, ymax) in EPSG:2056.

        Returns:
            tuple: (lon_min, lat_min, lon_max, lat_max) in EPSG:4326.
        """
        try:
            from pyproj import Transformer
            t = Transformer.from_crs("EPSG:2056", "EPSG:4326", always_xy=True)
            lon_min, lat_min = t.transform(bbox_lv95[0], bbox_lv95[1])
            lon_max, lat_max = t.transform(bbox_lv95[2], bbox_lv95[3])
            return lon_min, lat_min, lon_max, lat_max
        except ImportError as exc:
            raise ImportError(
                "pyproj is required for bbox reprojection. It is bundled with "
                "QGIS; if running standalone install it with: pip install pyproj"
            ) from exc

    def _query_items(
        self,
        bbox_wgs84: tuple,
        cb: Callable[[str], None],
    ) -> list:
        """Query STAC items intersecting *bbox_wgs84*, following pagination.

        Endpoint: GET {base_url}/collections/{collection}/items
        Parameters: bbox={w},{s},{e},{n}&limit={limit}
        Pagination: follow links[rel="next"] until absent.

        Args:
            bbox_wgs84: (lon_min, lat_min, lon_max, lat_max) in EPSG:4326.
            cb:         Progress callback.

        Returns:
            list[dict]: All STAC item dicts across pages.
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
            cb(f"  Fetching page {page}…")
            data = self._get_json(url)
            items.extend(data.get("features", []))

            # Follow the "next" link for pagination
            url = next(
                (lnk["href"] for lnk in data.get("links", []) if lnk.get("rel") == "next"),
                None,
            )
            page += 1

        return items

    def _deduplicate_latest(
        self,
        items: list,
        cb: Callable[[str], None],
    ) -> list:
        """Keep only the most recent acquisition per spatial cell.

        swissALTI3D item ids follow ``swissalti3d_<year>_<col>-<row>``; the
        ``<col>-<row>`` suffix identifies the 1 km² cell, which is repeated
        once per acquisition year. Items are grouped by that cell key and the
        one with the latest ``properties.datetime`` (falling back to the year
        embedded in the id) is kept.

        Args:
            items: STAC item dicts.
            cb:    Progress callback.

        Returns:
            list[dict]: One item per cell, newest acquisition first dropped of
            older duplicates. Order is otherwise preserved.
        """
        def cell_key(item: dict) -> str:
            parts = item.get("id", "").split("_")
            return parts[-1] if parts else item.get("id", "")

        def sort_value(item: dict) -> str:
            dt = item.get("properties", {}).get("datetime")
            if dt:
                return dt
            parts = item.get("id", "").split("_")
            return parts[1] if len(parts) > 2 else ""

        best: dict = {}
        for item in items:
            key = cell_key(item)
            if key not in best or sort_value(item) > sort_value(best[key]):
                best[key] = item

        dropped = len(items) - len(best)
        if dropped:
            cb(f"  Dropped {dropped} older duplicate tile(s).")
        return list(best.values())

    def _get_json(self, url: str) -> dict:
        """Perform a GET request and return the parsed JSON body.

        Args:
            url: Full URL to fetch.

        Returns:
            dict: Parsed JSON response.

        Raises:
            RuntimeError: On HTTP errors or JSON parse failures.
        """
        req = urllib.request.Request(url, headers={"User-Agent": _UA})
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f"STAC HTTP {exc.code} for {url}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"STAC network error: {exc.reason}") from exc
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Invalid JSON from {url}: {exc}") from exc

    # ── Private: asset selection ──────────────────────────────────────────────

    def _select_asset_href(self, item: dict) -> Optional[str]:
        """Pick the best GeoTIFF asset from a STAC item.

        Preference order:
          1. `proj:epsg` == target_epsg (2056)
          2. `eo:gsd` closest to target_resolution_m (2.0 m)
          3. Any GeoTIFF if no EPSG/GSD metadata is present (fallback)

        The swisstopo STAC API (v0.9) stores per-asset metadata directly in
        the asset object: {"href": "...", "type": "image/tiff; ...",
        "eo:gsd": 2.0, "proj:epsg": 2056, "checksum:multihash": "..."}.

        Args:
            item: STAC item dict (a GeoJSON Feature).

        Returns:
            str | None: href of the chosen asset, or None if no GeoTIFF found.
        """
        target_epsg = self._cfg["stac"].get("target_epsg", 2056)
        target_res  = self._cfg["stac"].get("target_resolution_m", 2.0)

        candidates = []
        for key, asset in item.get("assets", {}).items():
            mime = asset.get("type", "")
            if not mime.startswith("image/tiff"):
                continue
            epsg = asset.get("proj:epsg")
            gsd  = asset.get("eo:gsd")

            epsg_ok   = (epsg == target_epsg)
            res_score = abs(gsd - target_res) if isinstance(gsd, (int, float)) else 999.0

            # Sort key: (epsg_mismatch, res_delta) — lower is better
            candidates.append((0 if epsg_ok else 1, res_score, asset["href"]))

        if not candidates:
            return None

        candidates.sort()
        return candidates[0][2]   # href of the best candidate

    # ── Private: download ─────────────────────────────────────────────────────

    def _download_tiles(
        self,
        hrefs: list,
        cb: Callable[[str], None],
    ) -> list:
        """Download GeoTIFF tiles sequentially with progress reporting.

        Args:
            hrefs: List of download URLs.
            cb:    Progress callback.

        Returns:
            list[str]: Absolute paths to the downloaded tile files.

        Raises:
            RuntimeError: If any single download fails.
        """
        paths = []
        for i, href in enumerate(hrefs, 1):
            filename = os.path.basename(href.split("?")[0]) or f"tile_{i}.tif"
            dest     = os.path.join(self._work_dir, filename)

            if os.path.exists(dest):
                cb(f"  [{i}/{len(hrefs)}] Cached: {filename}")
                paths.append(dest)
                continue

            cb(f"  [{i}/{len(hrefs)}] Downloading {filename}…")
            req = urllib.request.Request(href, headers={"User-Agent": _UA})
            try:
                with urllib.request.urlopen(req, timeout=120) as resp, \
                     open(dest, "wb") as fh:
                    while chunk := resp.read(1 << 20):   # 1 MB chunks
                        fh.write(chunk)
            except (urllib.error.URLError, OSError) as exc:
                raise RuntimeError(f"Download failed for {href}: {exc}") from exc

            paths.append(dest)

        return paths

    def _ensure_crs(self, path: str, cb: Callable[[str], None]) -> None:
        """Re-tag *path* with the configured target EPSG if it is missing.

        swissALTI3D GeoTIFFs store their CRS as a ``LOCAL_CS`` WKT that carries
        ``AUTHORITY["EPSG","2056"]`` but is not recognised as a standard
        projected CRS, so ``crs.to_epsg()`` returns ``None``. This rewrites the
        CRS tag in place (cheap — only GeoTIFF header tags change) so downstream
        outputs inherit a clean, recognised EPSG code.

        Args:
            path: GeoTIFF whose CRS tag should be normalised.
            cb:   Progress callback.
        """
        target_epsg = self._cfg["stac"].get("target_epsg", 2056)
        try:
            import rasterio
            from rasterio.crs import CRS

            with rasterio.open(path) as src:
                current = src.crs.to_epsg() if src.crs else None
            if current == target_epsg:
                return

            with rasterio.open(path, "r+") as dst:
                dst.crs = CRS.from_epsg(target_epsg)
            cb(f"  CRS normalised to EPSG:{target_epsg}.")
        except ImportError as exc:
            raise ImportError(
                "rasterio is required to normalise the DTM CRS. "
                "It is bundled with QGIS."
            ) from exc

    # ── Private: mosaic ───────────────────────────────────────────────────────

    def _mosaic(self, tile_paths: list, cb: Callable[[str], None]) -> str:
        """Merge tiles into a single GeoTIFF using rasterio.merge.

        If only one tile was downloaded the file is returned directly
        (no merge needed).

        Args:
            tile_paths: Paths to downloaded tile GeoTIFFs.
            cb:         Progress callback.

        Returns:
            str: Absolute path to the output GeoTIFF.
        """
        if len(tile_paths) == 1:
            cb("  Single tile — skipping mosaic step.")
            return tile_paths[0]

        cb(f"  Mosaicking {len(tile_paths)} tiles…")
        mosaic_path = os.path.join(self._work_dir, "dtm_mosaic.tif")

        try:
            import rasterio
            from rasterio.merge import merge

            datasets = [rasterio.open(p) for p in tile_paths]
            try:
                mosaic_arr, mosaic_transform = merge(datasets)
                out_profile = datasets[0].profile.copy()
                out_profile.update(
                    height=mosaic_arr.shape[1],
                    width=mosaic_arr.shape[2],
                    transform=mosaic_transform,
                    compress="lzw",
                    tiled=True,
                )
                with rasterio.open(mosaic_path, "w", **out_profile) as dst:
                    dst.write(mosaic_arr)
            finally:
                for ds in datasets:
                    ds.close()

        except ImportError as exc:
            raise ImportError(
                "rasterio is required for mosaicking. It is bundled with QGIS."
            ) from exc

        cb(f"  Mosaic written ({os.path.getsize(mosaic_path) // 1024} KB).")
        return mosaic_path
