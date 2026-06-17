"""Morphometry — terrain morphometric analysis using rasterio/numpy.

Ported and extended from the Sorte landslide project (app.py):
  - DTM reading pattern: adapted from load_arr() — rasterio, band 1,
    float32 cast, nodata → np.nan, pixel resolution from src.res[0].
  - I/O helpers: rasterio-based, consistent with app.py conventions.
  - Slope, curvature: Zevenbergen & Thorne (1987), vectorised numpy.
  - Flow accumulation: D8 (O'Callaghan & Mark 1984) via pysheds
    (depression filling + flat resolution + C/numba-accelerated routing).
  - TWI: ln(SCA / tan(slope)), same formula as Sorte susceptibility model.

Pure Python / rasterio / numpy — NO QGIS imports.
Progress is reported through an optional callback so this module
can be unit-tested without a running QGIS instance.
"""

import warnings
from typing import Callable, Optional

import numpy as np

try:
    import rasterio
except ImportError as exc:
    raise ImportError(
        "rasterio not found. Install it (e.g. pip install rasterio) "
        "or use the OSGeo4W shell where it is bundled with QGIS."
    ) from exc


# ── Performance threshold ────────────────────────────────────────────────────
# Flow accumulation runs through pysheds (C/numba-accelerated), so even very
# large rasters are fast. Beyond this cell count we still emit a heads-up about
# memory footprint — pysheds holds several float/int copies of the grid in RAM.
_D8_CELL_WARN = 50_000_000   # ~200 km² at 2 m resolution


class Morphometry:
    """Compute slope (°), plan curvature (1/m) and TWI from a GeoTIFF DTM.

    All inputs and outputs are expected to be in EPSG:2056 (CH1903+ / LV95),
    a projected CRS where pixel resolution is in metres.
    """

    def __init__(self, dtm_path: str) -> None:
        """
        Args:
            dtm_path: Absolute path to the input DTM GeoTIFF (EPSG:2056, metres).
        """
        self._dtm_path = dtm_path

    # ── Public API ────────────────────────────────────────────────────────────

    def run(
        self,
        output_slope: str,
        output_twi: str,
        output_curvature: str,
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> None:
        """Compute and write slope, TWI and plan curvature rasters.

        Reads the DTM once, computes all three products in sequence, and
        writes them as Float32 GeoTIFFs in the same CRS, transform and
        extent as the input.

        Args:
            output_slope:      Absolute path for slope output GeoTIFF (°).
            output_twi:        Absolute path for TWI output GeoTIFF.
            output_curvature:  Absolute path for plan curvature GeoTIFF (1/m).
            progress_callback: Optional callable(str) for progress messages.

        Returns:
            None

        Raises:
            FileNotFoundError: If the DTM path does not exist.
            RuntimeError:      If rasterio cannot open the file.
        """
        cb = progress_callback or (lambda _: None)

        # ── 1. Read DTM ───────────────────────────────────────────────────────
        # Ported from load_arr() in app.py:
        #   rasterio.open → read(1).astype(float32) → nodata→nan
        #   res = abs(src.res[0])  (metres in LV95, no geo-correction needed)
        cb("Reading DTM…")
        dem, res, meta = self._read_dtm()
        cb(f"DTM loaded — {dem.shape[1]}×{dem.shape[0]} px, res={res:.2f} m")

        # Output profile: same CRS/transform as input, Float32, LZW compressed
        out_profile = meta.copy()
        out_profile.update(
            dtype="float32",
            nodata=-9999.0,
            count=1,
            compress="lzw",
            tiled=True,
        )

        # ── 2. Slope ──────────────────────────────────────────────────────────
        cb("Computing slope (°)…")
        slope_deg = self._compute_slope(dem, res)
        self._write(output_slope, slope_deg, out_profile)
        cb("  → slope written.")

        # ── 3. Plan curvature ─────────────────────────────────────────────────
        cb("Computing plan curvature (1/m)…")
        curv = self._compute_plan_curvature(dem, res)
        self._write(output_curvature, curv, out_profile)
        cb("  → curvature written.")

        # ── 4. Flow accumulation + TWI ────────────────────────────────────────
        n_cells = dem.shape[0] * dem.shape[1]
        if n_cells > _D8_CELL_WARN:
            cb(
                f"  ⚠ Very large raster ({n_cells:,} cells). pysheds keeps "
                "several full-grid copies in memory; ensure enough RAM is "
                "available or clip the AOI."
            )

        cb("Computing D8 flow accumulation…")
        sca = self._compute_flow_accumulation(dem, res, cb)

        cb("Computing TWI…")
        slope_rad = np.deg2rad(slope_deg)
        twi = self._compute_twi(slope_rad, sca)
        self._write(output_twi, twi, out_profile)
        cb("  → TWI written.")

        cb("Morphometry complete.")

    # ── Private: I/O ─────────────────────────────────────────────────────────

    def _read_dtm(self) -> tuple[np.ndarray, float, dict]:
        """Read the DTM GeoTIFF and return (array, resolution_m, rasterio_profile).

        Adapted from load_arr() in app.py:
            arr = src.read(1).astype(np.float32)
            if nd is not None: arr[arr == nd] = np.nan
            res = abs(src.res[0])   # metres in LV95 (projected CRS)

        Returns:
            dem:  2D float32 array, NoData cells set to np.nan.
            res:  Pixel size in metres.
            meta: rasterio profile dict (driver, crs, transform, width, height …).

        Raises:
            FileNotFoundError: If self._dtm_path does not exist.
        """
        import os
        if not os.path.exists(self._dtm_path):
            raise FileNotFoundError(f"DTM not found: {self._dtm_path}")

        with rasterio.open(self._dtm_path) as src:
            if src.crs and src.crs.is_geographic:
                raise RuntimeError(
                    f"DTM CRS is geographic (EPSG:{src.crs.to_epsg()}). "
                    "The morphometry formulas require a metric projected CRS. "
                    "Reproject the DTM to EPSG:2056 (or any other metric CRS) "
                    "before running this tool."
                )
            dem  = src.read(1).astype(np.float32)
            meta = src.profile.copy()
            nd   = src.nodata
            res  = abs(src.res[0])   # metres in any metric projected CRS

        if nd is not None:
            dem[dem == nd] = np.nan

        return dem, res, meta

    def _write(self, path: str, array: np.ndarray, profile: dict) -> None:
        """Write a 2D float32 array as a GeoTIFF using rasterio.

        NoData value is taken from profile['nodata']; np.nan cells are
        replaced with it before writing.

        Args:
            path:    Destination file path.
            array:   2D float32 array to write.
            profile: rasterio profile dict (must include crs, transform, nodata …).
        """
        nd  = float(profile["nodata"])
        arr = np.where(np.isnan(array), nd, array).astype(np.float32)
        with rasterio.open(path, "w", **profile) as dst:
            dst.write(arr, 1)

    # ── Private: morphometry computations ────────────────────────────────────

    def _compute_slope(self, dem: np.ndarray, res: float) -> np.ndarray:
        """Compute slope in degrees using Zevenbergen & Thorne (1987).

        Uses a 3×3 moving window with edge-padded borders.
        G = ∂z/∂x (eastward),  H = ∂z/∂y (northward).
        slope = arctan(√(G² + H²)).

        Args:
            dem: 2D float32 array. NoData cells must be np.nan.
            res: Pixel resolution in metres.

        Returns:
            2D float32 array of slope values in degrees. np.nan where dem is nan.
        """
        p = np.pad(dem, 1, mode="edge")   # edge-pad to handle borders cleanly

        # Cardinal neighbours (raster convention: row 0 = north)
        z_w = p[1:-1, :-2]   # west
        z_e = p[1:-1,  2:]   # east
        z_n = p[:-2, 1:-1]   # north (smaller row index = higher latitude in LV95)
        z_s = p[2:,  1:-1]   # south

        G = (z_e - z_w) / (2.0 * res)   # ∂z/∂x
        H = (z_n - z_s) / (2.0 * res)   # ∂z/∂y

        slope = np.degrees(np.arctan(np.sqrt(G**2 + H**2))).astype(np.float32)
        slope[np.isnan(dem)] = np.nan
        return slope

    def _compute_plan_curvature(self, dem: np.ndarray, res: float) -> np.ndarray:
        """Compute plan curvature (1/m) using Zevenbergen & Thorne (1987).

        Negative values = concave (convergent flow).
        Positive values = convex (divergent flow).
        Formula: kplan = -2(D·H² + E·G² - F·G·H) / (G² + H²)
        where D=∂²z/∂x², E=∂²z/∂y², F=∂²z/∂x∂y, G=∂z/∂x, H=∂z/∂y.
        Returns 0.0 on flat terrain (|∇z|² < 1e-8).

        Args:
            dem: 2D float32 array. NoData = np.nan.
            res: Pixel resolution in metres.

        Returns:
            2D float32 plan curvature array (1/m). np.nan where dem is nan.
        """
        p = np.pad(dem, 1, mode="edge")

        z5  = p[1:-1, 1:-1]   # centre
        z_w = p[1:-1, :-2]    # west
        z_e = p[1:-1,  2:]    # east
        z_n = p[:-2, 1:-1]    # north
        z_s = p[2:,  1:-1]    # south
        z_nw = p[:-2, :-2]    # NW
        z_ne = p[:-2,  2:]    # NE
        z_sw = p[2:,  :-2]    # SW
        z_se = p[2:,   2:]    # SE

        r2 = res ** 2

        D = ((z_w + z_e) / 2.0 - z5) / r2              # ∂²z/∂x²
        E = ((z_n + z_s) / 2.0 - z5) / r2              # ∂²z/∂y²
        F = (-z_nw + z_ne + z_sw - z_se) / (4.0 * r2)  # ∂²z/∂x∂y
        G = (z_e - z_w) / (2.0 * res)                  # ∂z/∂x
        H = (z_n - z_s) / (2.0 * res)                  # ∂z/∂y

        sq_grad = G**2 + H**2   # |∇z|²

        with np.errstate(invalid="ignore", divide="ignore"):
            kplan = np.where(
                sq_grad > 1e-8,
                -2.0 * (D * H**2 + E * G**2 - F * G * H) / sq_grad,
                0.0,
            ).astype(np.float32)

        kplan[np.isnan(dem)] = np.nan
        return kplan

    def _compute_flow_accumulation(
        self,
        dem: np.ndarray,
        res: float,
        cb: Callable[[str], None],
    ) -> np.ndarray:
        """Compute specific catchment area (m²/m) via D8 flow routing.

        Delegates the heavy lifting to pysheds, whose routing is implemented
        in C / numba-compiled kernels — no Python per-cell loop — so it scales
        to 100M-cell rasters that the previous pure-Python accumulation could
        not handle in reasonable time.

        Pipeline (O'Callaghan & Mark 1984 D8, the standard pysheds sequence):
          1. ``fill_depressions`` — remove single-cell and multi-cell pits.
          2. ``resolve_flats``    — impose a drainage gradient on flat areas.
          3. ``flowdir``          — D8 flow direction (ESRI dirmap encoding).
          4. ``accumulation``     — number of upstream cells (incl. the cell
             itself), each weighted by 1.
          5. SCA = n_cells · res² / res = n_cells · res  (m²/m, unit-width
             catchment area), identical in meaning to the previous output.

        NoData handling: np.nan cells are passed to pysheds as a sentinel
        NoData value and restored to np.nan in the result.

        Args:
            dem: 2D float32 array. NoData = np.nan.
            res: Pixel resolution in metres.
            cb:  Progress callback for step messages.

        Returns:
            2D float32 SCA array (m²/m). np.nan where dem is nan.
        """
        try:
            from pysheds.grid import Grid
            from pysheds.view import Raster, ViewFinder
        except ImportError as exc:
            raise ImportError(
                "pysheds not found. Install it (e.g. pip install pysheds) "
                "to compute D8 flow accumulation. In the OSGeo4W / QGIS "
                "Python shell run: python -m pip install pysheds"
            ) from exc

        # pysheds 0.5 still calls np.in1d, removed in NumPy 2.0. Restore the
        # alias when running against NumPy ≥ 2 so the plugin works on both.
        if not hasattr(np, "in1d"):
            np.in1d = np.isin   # harmless: re-adds a removed public alias

        nan_mask = np.isnan(dem)

        # ── Wrap the array as a pysheds Raster with an explicit NoData ────
        nodata = np.float64(-9999.0)
        dem_filled = np.where(nan_mask, nodata, dem).astype(np.float64)
        viewfinder = ViewFinder(shape=dem.shape, nodata=nodata)
        dem_raster = Raster(dem_filled, viewfinder)
        grid = Grid(viewfinder=viewfinder)

        # ESRI D8 direction encoding (N, NE, E, SE, S, SW, W, NW)
        dirmap = (64, 128, 1, 2, 4, 8, 16, 32)

        cb("  D8 step 1/3 — filling depressions & resolving flats…")
        flooded = grid.fill_depressions(dem_raster)
        inflated = grid.resolve_flats(flooded)

        cb("  D8 step 2/3 — computing flow direction…")
        fdir = grid.flowdir(inflated, dirmap=dirmap, nodata_out=np.int64(0))

        cb("  D8 step 3/3 — accumulating flow (pysheds, vectorised)…")
        acc = grid.accumulation(fdir, dirmap=dirmap)

        # acc = upstream cell count (incl. self); area = acc·res²,
        # SCA = area / res = acc · res  (m²/m, unit-width).
        sca = (np.asarray(acc, dtype=np.float64) * res).astype(np.float32)
        sca[nan_mask] = np.nan
        return sca

    def _compute_twi(
        self,
        slope_rad: np.ndarray,
        sca: np.ndarray,
    ) -> np.ndarray:
        """Compute TWI = ln(SCA / tan(slope)).

        Same formula used in the Sorte landslide susceptibility model.
        Minimum slope clamped to 1e-6 rad to avoid division by zero on
        flat terrain. SCA values below 1.0 m²/m are clamped to 1.0.

        Args:
            slope_rad: 2D slope array in radians.
            sca:       2D specific catchment area array (m²/m).

        Returns:
            2D float32 TWI array. np.nan where either input is nan.
        """
        nan_mask = np.isnan(slope_rad) | np.isnan(sca)

        tan_sl   = np.tan(np.where(slope_rad < 1e-6, 1e-6, slope_rad))
        safe_sca = np.where(np.isnan(sca) | (sca < 1.0), 1.0, sca)

        with np.errstate(invalid="ignore", divide="ignore"):
            twi = np.log(safe_sca / tan_sl).astype(np.float32)

        twi[nan_mask] = np.nan
        return twi
