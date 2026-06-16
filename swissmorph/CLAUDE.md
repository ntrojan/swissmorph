# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this plugin does

SwissMorph is a QGIS 3 Processing plugin. Given an area of interest in Switzerland it:
1. Downloads swissALTI3D 2 m DTM tiles from the swisstopo STAC API (`data.geo.admin.ch/api/stac/v0.9`, collection `ch.swisstopo.swissalti3d`).
2. Mosaics them into a single GeoTIFF.
3. Derives seven morphometric layers — all in EPSG:2056 (CH1903+/LV95):
   - Slope (°)
   - Plan curvature (1/m) — Zevenbergen & Thorne (1987)
   - Profile curvature (1/m) — Zevenbergen & Thorne (1987)
   - TWI (Topographic Wetness Index)
   - SPI (Stream Power Index) = SCA × tan(slope)
   - LS factor (RUSLE) — Moore & Burch (1986)
   - TPI (Topographic Position Index) — Weiss (2001), 3×3 window

## Running tests (no QGIS needed)

`core/` and `tests/` are QGIS-agnostic and run with plain Python:

```
# from the parent directory of swissmorph/ (i.e. one level up from this file)
python -m pytest swissmorph/tests/ -v
# or with unittest
python -m unittest discover -s swissmorph/tests -v
```

Run a single test file:
```
python -m pytest swissmorph/tests/test_morphometry.py -v
python -m pytest swissmorph/tests/test_stac.py -v
```

Dependencies required: `rasterio`, `numpy`, `pyproj`. These are bundled inside QGIS/OSGeo4W. For standalone test runs outside QGIS: `pip install rasterio numpy pyproj`.

## Architecture

The codebase has a strict two-layer split enforced deliberately:

```
swissmorph/
├── core/              # Pure Python — NO QGIS imports allowed
│   ├── morphometry.py # Morphometry class: slope/curvature/TWI via numpy
│   └── stac.py        # StacDownloader: STAC query, tile download, mosaic
├── algorithms/
│   └── morphometry_algorithm.py  # MorphometryAlgorithm (QgsProcessingAlgorithm)
├── plugin.py          # SwissMorphPlugin: QGIS lifecycle (initGui / unload)
├── provider.py        # SwissMorphProvider: registers algorithms in Processing Toolbox
├── config/
│   └── defaults.json  # STAC endpoint, target EPSG and resolution, cache prefix
└── __init__.py        # classFactory — deferred import so core/ is testable standalone
```

**Layer boundary rule**: `core/` must never import from `qgis`. All QGIS types (`QgsRectangle`, `QgsCoordinateTransform`, etc.) are handled in `algorithms/morphometry_algorithm.py`, which then passes plain Python values (tuples, file paths, callbacks) into `core/`.

## Key design details

**AOI inputs**: polygon layer (highest priority) or map extent. The extent widget has a working "Select on canvas" button for drawing rectangles directly on the map. The old `QgsProcessingParameterGeometry` (free-draw) was removed — it had no canvas tool in QGIS Processing dialogs.

**D8 flow accumulation** (`Morphometry._compute_flow_accumulation`) is a pure Python O(N) loop with cancellation checks every `max(100 000, N/50)` iterations. It warns via the progress callback when the raster exceeds `_D8_CELL_WARN = 4_000_000` cells (~4 km² at 2 m). For larger AOIs, WhiteboxTools is the intended replacement.

**CRS flow**: AOI arrives in any CRS → reprojected to EPSG:2056 in `_resolve_aoi` → `bbox_lv95` tuple passed to `StacDownloader.fetch()` → reprojected to EPSG:4326 inside `StacDownloader._to_wgs84()` for the STAC query → tiles downloaded and mosaicked in EPSG:2056 → morphometry computed in metres.

**Temp dir cleanup**: `processAlgorithm` wraps the download + morphometry in `try/finally` and calls `shutil.rmtree(tmp_dir, ignore_errors=True)` on exit. Output rasters are written to QGIS-managed paths (not inside tmp_dir), so cleanup is safe.

**`Morphometry.run()` returns** `bool` — `True` if completed, `False` if cancelled. Accepts `cancel_callback` (forwarded to the D8 loop) and `progress_setter` (mapped from 0-100 % to 30-100 % in the algorithm).

**Shared curvature helper**: `_curvature_components(dem, res)` computes the five Zevenbergen & Thorne partial derivatives (D, E, F, G, H) once; both `_compute_plan_curvature` and `_compute_profile_curvature` call it independently.

**Config**: `config/defaults.json` is loaded relative to `core/stac.py`. A `config` dict can be injected into `StacDownloader.__init__` to override it (used in tests).

## Installing / reloading in QGIS

The plugin folder is already in the QGIS plugin path. To reload after code changes, use the QGIS Plugin Reloader plugin or restart QGIS. There is no build step.
