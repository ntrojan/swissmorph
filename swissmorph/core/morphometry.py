"""Morphometry — terrain morphometric analysis using rasterio/numpy.

Algorithms:
  - Slope, plan curvature, profile curvature: Zevenbergen & Thorne (1987).
  - Flow accumulation (SCA): D8, O'Callaghan & Mark (1984).
  - TWI: ln(SCA / tan(slope)).
  - SPI: SCA × tan(slope).
  - LS factor: Moore & Burch (1986) — (SCA/22.13)^0.4 × (sin(slope)/0.0896)^1.3.
  - TPI: elevation minus 3×3 neighbourhood mean (Weiss 2001).

Pure Python / rasterio / numpy — NO QGIS imports.
"""

import math
from typing import Callable, Optional

import numpy as np

try:
    import rasterio
except ImportError as exc:
    raise ImportError(
        "rasterio not found. Install it (e.g. pip install rasterio) "
        "or use the OSGeo4W shell where it is bundled with QGIS."
    ) from exc


# D8 accumulation loop is O(N) Python iterations — warn above this threshold.
_D8_CELL_WARN = 4_000_000   # ~4 km² at 2 m resolution

# None=unchecked, False=not found, str=absolute path to whitebox_tools binary.
_WBT_EXE = None


def _safe_listdir(path: str) -> list:
    import os
    try:
        return os.listdir(path)
    except OSError:
        return []


def _find_wbt_exe() -> "str | None":
    """Locate whitebox_tools binary. Search order:
    1. Python 'whitebox' package (pip install whitebox)
    2. QGIS plugin directories in the user profile (covers 'WhiteboxTools for QGIS')
    3. System PATH
    """
    import os, sys, shutil

    exe_name = "whitebox_tools.exe" if sys.platform == "win32" else "whitebox_tools"

    # 1. Python whitebox package — exe_path may be the binary or its parent directory
    try:
        import whitebox as _wb
        exe_dir_or_file = _wb.WhiteboxTools().exe_path
        if os.path.isfile(exe_dir_or_file):
            return exe_dir_or_file
        candidate = os.path.join(exe_dir_or_file, exe_name)
        if os.path.isfile(candidate):
            return candidate
    except Exception:
        pass

    # 2. QGIS user plugin directories
    if sys.platform == "win32":
        roots = [os.path.join(os.environ.get("APPDATA", ""), "QGIS", "QGIS3", "profiles")]
    else:
        home = os.path.expanduser("~")
        roots = [
            os.path.join(home, ".local", "share", "QGIS", "QGIS3", "profiles"),
            os.path.join(home, "Library", "Application Support", "QGIS", "QGIS3", "profiles"),
        ]

    for profiles_root in roots:
        if not os.path.isdir(profiles_root):
            continue
        for profile in _safe_listdir(profiles_root):
            plugins_dir = os.path.join(profiles_root, profile, "python", "plugins")
            if not os.path.isdir(plugins_dir):
                continue
            for plugin_name in _safe_listdir(plugins_dir):
                plugin_dir = os.path.join(plugins_dir, plugin_name)
                if not os.path.isdir(plugin_dir):
                    continue
                for root, dirs, files in os.walk(plugin_dir):
                    if exe_name in files:
                        return os.path.join(root, exe_name)
                    if root[len(plugin_dir):].count(os.sep) >= 2:
                        dirs[:] = []  # don't descend more than 2 levels

    # 3. System PATH
    return shutil.which(exe_name)


def set_wbt_exe_hint(path: str) -> None:
    """Pre-seed the WBT exe path from the algorithm layer (e.g. from ProcessingConfig).

    Call this before the first run() if you already know the binary location.
    Has no effect if the path is empty, does not exist, or WBT was already found.
    """
    import os
    global _WBT_EXE
    if _WBT_EXE is None and path and os.path.isfile(path):
        _WBT_EXE = path


def _get_wbt_exe(cb) -> "str | None":
    """Return cached whitebox_tools exe path, searching on first call."""
    global _WBT_EXE
    if _WBT_EXE is not None:
        return _WBT_EXE if _WBT_EXE else None
    exe = _find_wbt_exe()
    _WBT_EXE = exe or False
    if exe:
        cb(f"  WhiteboxTools found: {exe}")
    else:
        cb(
            "  WhiteboxTools binary not found. "
            "The 'wbt_for_qgis' plugin requires the WBT binary to be downloaded separately. "
            "Download WhiteboxTools, extract it, then set the path in: "
            "QGIS Settings → Options → Processing → Providers → WhiteboxTools "
            "→ WhiteboxTools executable."
        )
    return exe

# LS factor constants (Moore & Burch 1986 / RUSLE)
_LS_M              = 0.4      # slope-length exponent
_LS_N              = 1.3      # slope-steepness exponent
_LS_PLOT_LENGTH    = 22.13    # m  (USLE unit-plot length)
_LS_PLOT_SLOPE_SIN = 0.0896   # sin(5.143°)  (USLE unit-plot slope)


class Morphometry:
    """Compute terrain derivatives from a GeoTIFF DTM.

    All inputs/outputs are expected in a metric projected CRS (e.g. EPSG:2056).
    """

    def __init__(self, dtm_path: str) -> None:
        self._dtm_path = dtm_path

    # ---Public API ────────────────────────────────────────────────────────────

    def run(
        self,
        output_slope: str,
        output_twi: str,
        output_curvature: str,
        output_profile_curvature: str,
        output_spi: str,
        output_ls_factor: str,
        output_tpi: str,
        tpi_radius: int = 1,
        progress_callback: Optional[Callable[[str], None]] = None,
        cancel_callback: Optional[Callable[[], bool]] = None,
        progress_setter: Optional[Callable[[int], None]] = None,
    ) -> bool:
        """Compute and write all seven morphometric products.

        Args:
            output_slope:            Path for slope GeoTIFF (°).
            output_twi:              Path for TWI GeoTIFF.
            output_curvature:        Path for plan curvature GeoTIFF (1/m).
            output_profile_curvature: Path for profile curvature GeoTIFF (1/m).
            output_spi:              Path for Stream Power Index GeoTIFF.
            output_ls_factor:        Path for LS factor GeoTIFF.
            output_tpi:              Path for TPI GeoTIFF (m).
            tpi_radius:              Neighbourhood radius in cells (default 1 → 3×3 window).
            progress_callback:       Optional callable(str) for log messages.
            cancel_callback:         Optional callable() → bool; True = cancel.
            progress_setter:         Optional callable(int) for 0-100 progress.

        Returns:
            True if completed; False if cancelled.
        """
        cb      = progress_callback or (lambda _: None)
        set_pct = progress_setter   or (lambda _: None)

        # ---1. Read DTM ───────────────────────────────────────────────────────
        cb("Reading DTM…")
        dem, res, meta = self._read_dtm()
        cb(f"DTM loaded — {dem.shape[1]}×{dem.shape[0]} px, res={res:.2f} m")
        set_pct(5)

        out_profile = meta.copy()
        out_profile.update(
            dtype="float32",
            nodata=-9999.0,
            count=1,
            compress="lzw",
            tiled=True,
        )

        # ---2. Slope ──────────────────────────────────────────────────────────
        cb("Computing slope (°)…")
        slope_deg = self._compute_slope(dem, res)
        slope_rad = np.deg2rad(slope_deg)
        self._write(output_slope, slope_deg, out_profile)
        cb("  → slope written.")
        set_pct(15)

        # ---3. Plan curvature ─────────────────────────────────────────────────
        cb("Computing plan curvature (1/m)…")
        kplan = self._compute_plan_curvature(dem, res)
        self._write(output_curvature, kplan, out_profile)
        cb("  → plan curvature written.")
        set_pct(25)

        # ---4. Profile curvature ──────────────────────────────────────────────
        cb("Computing profile curvature (1/m)…")
        kprof = self._compute_profile_curvature(dem, res)
        self._write(output_profile_curvature, kprof, out_profile)
        cb("  → profile curvature written.")
        set_pct(35)

        # ---5. TPI ────────────────────────────────────────────────────────────
        r = max(1, int(tpi_radius))
        win = 2 * r + 1
        cb(f"Computing TPI ({win}×{win} window, radius={r} cells, m)…")
        tpi = self._compute_tpi(dem, radius=r)
        self._write(output_tpi, tpi, out_profile)
        cb("  → TPI written.")
        set_pct(40)

        # ---6. D8 flow accumulation ───────────────────────────────────────────
        cb("Computing D8 flow accumulation…")
        sca = self._compute_flow_accumulation(dem, res, cb, cancel_callback)
        if sca is None:
            cb("Cancelled during flow accumulation.")
            return False
        set_pct(80)

        # ---7. TWI ────────────────────────────────────────────────────────────
        cb("Computing TWI…")
        twi = self._compute_twi(slope_rad, sca)
        self._write(output_twi, twi, out_profile)
        cb("  → TWI written.")
        set_pct(85)

        # ---8. SPI ────────────────────────────────────────────────────────────
        cb("Computing SPI…")
        spi = self._compute_spi(slope_rad, sca)
        self._write(output_spi, spi, out_profile)
        cb("  → SPI written.")
        set_pct(90)

        # ---9. LS factor ──────────────────────────────────────────────────────
        cb("Computing LS factor…")
        ls = self._compute_ls_factor(slope_rad, sca)
        self._write(output_ls_factor, ls, out_profile)
        cb("  → LS factor written.")
        set_pct(95)

        cb("Morphometry complete.")
        return True

    # ---Private: I/O ─────────────────────────────────────────────────────────

    def _read_dtm(self) -> tuple:
        """Read the DTM GeoTIFF → (array float32, resolution_m, rasterio_profile).

        Raises:
            FileNotFoundError: If self._dtm_path does not exist.
            RuntimeError:      If the CRS is geographic (metres required).
        """
        import os
        if not os.path.exists(self._dtm_path):
            raise FileNotFoundError(f"DTM not found: {self._dtm_path}")

        with rasterio.open(self._dtm_path) as src:
            if src.crs and src.crs.is_geographic:
                raise RuntimeError(
                    f"DTM CRS is geographic (EPSG:{src.crs.to_epsg()}). "
                    "The morphometry formulas require a metric projected CRS. "
                    "Reproject the DTM to EPSG:2056 before running this tool."
                )
            dem  = src.read(1).astype(np.float32)
            meta = src.profile.copy()
            nd   = src.nodata
            res  = abs(src.res[0])

        if nd is not None:
            dem[dem == nd] = np.nan

        return dem, res, meta

    def _write(self, path: str, array: np.ndarray, profile: dict) -> None:
        """Write a 2D float32 array as a GeoTIFF; np.nan → profile nodata."""
        nd  = float(profile["nodata"])
        arr = np.where(np.isnan(array), nd, array).astype(np.float32)
        with rasterio.open(path, "w", **profile) as dst:
            dst.write(arr, 1)

    # ---Private: shared gradient helper ──────────────────────────────────────

    def _curvature_components(
        self, dem: np.ndarray, res: float
    ) -> tuple:
        """Zevenbergen & Thorne (1987) partial derivatives for a 3×3 window.

        Returns:
            (D, E, F, G, H, sq_grad) where
              D = ∂²z/∂x²,  E = ∂²z/∂y²,  F = ∂²z/∂x∂y,
              G = ∂z/∂x,    H = ∂z/∂y,
              sq_grad = G² + H²
        """
        p    = np.pad(dem, 1, mode="edge")
        z5   = p[1:-1, 1:-1]
        z_w  = p[1:-1, :-2];  z_e  = p[1:-1,  2:]
        z_n  = p[:-2,  1:-1]; z_s  = p[2:,   1:-1]
        z_nw = p[:-2,  :-2];  z_ne = p[:-2,   2:]
        z_sw = p[2:,   :-2];  z_se = p[2:,    2:]
        r2   = res ** 2
        D = ((z_w + z_e) / 2.0 - z5) / r2
        E = ((z_n + z_s) / 2.0 - z5) / r2
        F = (-z_nw + z_ne + z_sw - z_se) / (4.0 * r2)
        G = (z_e - z_w) / (2.0 * res)
        H = (z_n - z_s) / (2.0 * res)
        return D, E, F, G, H, G ** 2 + H ** 2

    # ---Private: morphometry computations ────────────────────────────────────

    def _compute_slope(self, dem: np.ndarray, res: float) -> np.ndarray:
        """Slope in degrees — Zevenbergen & Thorne (1987).

        slope = arctan(√(G² + H²))
        """
        p   = np.pad(dem, 1, mode="edge")
        z_w = p[1:-1, :-2]; z_e = p[1:-1,  2:]
        z_n = p[:-2, 1:-1]; z_s = p[2:,  1:-1]
        G = (z_e - z_w) / (2.0 * res)
        H = (z_n - z_s) / (2.0 * res)
        slope = np.degrees(np.arctan(np.sqrt(G ** 2 + H ** 2))).astype(np.float32)
        slope[np.isnan(dem)] = np.nan
        return slope

    def _compute_plan_curvature(self, dem: np.ndarray, res: float) -> np.ndarray:
        """Plan curvature (1/m) — Zevenbergen & Thorne (1987).

        Negative = concave (convergent flow). Positive = convex (divergent flow).
        Formula: kplan = −2(D·H² + E·G² − F·G·H) / (G² + H²)
        Returns 0 on flat terrain (|∇z|² < 1e-8).
        """
        D, E, F, G, H, sq_grad = self._curvature_components(dem, res)
        with np.errstate(invalid="ignore", divide="ignore"):
            kplan = np.where(
                sq_grad > 1e-8,
                -2.0 * (D * H ** 2 + E * G ** 2 - F * G * H) / sq_grad,
                0.0,
            ).astype(np.float32)
        kplan[np.isnan(dem)] = np.nan
        return kplan

    def _compute_profile_curvature(self, dem: np.ndarray, res: float) -> np.ndarray:
        """Profile curvature (1/m) — Zevenbergen & Thorne (1987).

        Curvature measured in the slope direction (along flow).
        Negative = concave downslope (flow accelerates).
        Positive = convex downslope (flow decelerates).
        Formula: kprof = −2(D·G² + E·H² + F·G·H) / (G² + H²)
        Returns 0 on flat terrain (|∇z|² < 1e-8).
        """
        D, E, F, G, H, sq_grad = self._curvature_components(dem, res)
        with np.errstate(invalid="ignore", divide="ignore"):
            kprof = np.where(
                sq_grad > 1e-8,
                -2.0 * (D * G ** 2 + E * H ** 2 + F * G * H) / sq_grad,
                0.0,
            ).astype(np.float32)
        kprof[np.isnan(dem)] = np.nan
        return kprof

    def _compute_tpi(self, dem: np.ndarray, radius: int = 1) -> np.ndarray:
        """Topographic Position Index (m) — Weiss (2001), configurable window.

        TPI = z_centre − mean(neighbourhood)
        Neighbourhood is a (2*radius+1)×(2*radius+1) square excluding the centre.
        Positive = local high (ridge/peak). Negative = local low (valley/hollow).
        NaN cells are excluded from the neighbourhood mean via integral images.

        Uses a 2D integral image (prefix sum) so runtime is O(N) regardless of
        radius — no stacking of neighbour planes in memory.
        """
        r = max(1, int(radius))
        nan_mask = np.isnan(dem)
        fill  = np.where(nan_mask, 0.0, dem.astype(np.float64))
        valid = (~nan_mask).astype(np.float64)
        rows, cols = dem.shape
        win = 2 * r + 1

        # Pad with r+1 zeros on top/left and r zeros on bottom/right so that
        # the window for centre cell (i, j) maps cleanly to padded[i+1..i+win].
        p_fill  = np.pad(fill,  ((r + 1, r), (r + 1, r)), constant_values=0.0)
        p_valid = np.pad(valid, ((r + 1, r), (r + 1, r)), constant_values=0.0)

        def _integral(arr: np.ndarray) -> np.ndarray:
            h, w = arr.shape
            ii = np.zeros((h + 1, w + 1), dtype=np.float64)
            ii[1:, 1:] = np.cumsum(np.cumsum(arr, axis=0), axis=1)
            return ii

        ii_f = _integral(p_fill)
        ii_v = _integral(p_valid)

        # Window box-sum formula: II[r2+1,c2+1] - II[r1,c2+1] - II[r2+1,c1] + II[r1,c1]
        # For centre (i,j): window rows i+1..i+win, cols j+1..j+win  →  r1=i+1, r2=i+win
        ra = slice(win + 1, win + 1 + rows)   # ii row index r2+1
        rb = slice(1,       1 + rows)          # ii row index r1
        ca = slice(win + 1, win + 1 + cols)   # ii col index c2+1
        cb_ = slice(1,      1 + cols)          # ii col index c1

        win_f = ii_f[ra, ca] - ii_f[rb, ca] - ii_f[ra, cb_] + ii_f[rb, cb_]
        win_v = ii_v[ra, ca] - ii_v[rb, ca] - ii_v[ra, cb_] + ii_v[rb, cb_]

        # Exclude centre cell from the neighbourhood sum/count
        nb_sum   = win_f - fill
        nb_count = win_v - valid

        with np.errstate(invalid="ignore", divide="ignore"):
            nb_mean = np.where(nb_count > 0, nb_sum / nb_count, np.nan)

        tpi = (dem - nb_mean).astype(np.float32)
        tpi[np.isnan(dem)] = np.nan
        return tpi

    def _wbt_d8(self, cb: Callable[[str], None]) -> Optional[np.ndarray]:
        """Run D8 flow accumulation via the whitebox_tools binary (subprocess).

        Calls BreachDepressionsLeastCost then D8FlowAccumulation.
        Returns SCA as float32 (m²/m), or None if WBT is unavailable or fails.
        """
        import os, subprocess

        exe = _get_wbt_exe(cb)
        if exe is None:
            return None

        wbt_dir = os.path.dirname(self._dtm_path)
        filled  = os.path.join(wbt_dir, "dem_filled.tif")
        acc_out = os.path.join(wbt_dir, "acc_wbt.tif")

        try:
            cb("  D8 (WBT): breaching depressions…")
            r = subprocess.run(
                [exe, "--run=BreachDepressionsLeastCost",
                 f"--dem={self._dtm_path}", f"--output={filled}",
                 "--dist=5", "--fill"],
                capture_output=True, text=True, timeout=600,
            )
            if r.returncode != 0:
                raise RuntimeError(r.stderr.strip() or "BreachDepressionsLeastCost failed")

            cb("  D8 (WBT): accumulating flow…")
            r = subprocess.run(
                [exe, "--run=D8FlowAccumulation",
                 f"--input={filled}", f"--output={acc_out}",
                 "--out_type=specific contributing area"],
                capture_output=True, text=True, timeout=600,
            )
            if r.returncode != 0:
                raise RuntimeError(r.stderr.strip() or "D8FlowAccumulation failed")

            with rasterio.open(acc_out) as src:
                sca = src.read(1).astype(np.float32)
            cb("  D8 (WBT): done.")
            return sca

        except Exception as exc:
            cb(f"  WhiteboxTools D8 failed: {exc} — falling back to Python D8.")
            return None

    def _compute_flow_accumulation(
        self,
        dem: np.ndarray,
        res: float,
        cb: Callable[[str], None],
        cancel_callback: Optional[Callable[[], bool]] = None,
    ) -> Optional[np.ndarray]:
        """D8 specific catchment area (m²/m).

        For rasters above _D8_CELL_WARN cells, tries WhiteboxTools first
        (faster, includes sink filling). Falls back to the pure-Python
        O'Callaghan & Mark (1984) implementation if WBT is unavailable.
        Returns None if cancel_callback fires during the Python accumulation loop.
        """
        rows, cols = dem.shape
        n_cells = rows * cols

        if n_cells > _D8_CELL_WARN:
            cb(
                f"  ⚠ Large raster ({n_cells:,} cells). "
                "Trying WhiteboxTools for faster D8…"
            )
            sca = self._wbt_d8(cb)
            if sca is not None:
                return sca
            cb(
                f"  Using Python D8 — estimated {n_cells // 500_000:.0f}–"
                f"{n_cells // 200_000:.0f} s."
            )

        D8 = [(-1,-1), (-1, 0), (-1, 1),
              ( 0,-1),          ( 0, 1),
              ( 1,-1), ( 1, 0), ( 1, 1)]
        W  = np.array([math.sqrt(2), 1.0, math.sqrt(2),
                       1.0,               1.0,
                       math.sqrt(2), 1.0, math.sqrt(2)]) * res

        # Step 1: drop to each of the 8 neighbours
        cb("  D8 step 1/3 — computing drops…")
        drops = np.full((8, rows, cols), -np.inf, dtype=np.float32)
        for d, (di, dj) in enumerate(D8):
            r_s = slice(1, rows)   if di == -1 else (slice(0, rows-1) if di == 1 else slice(0, rows))
            r_n = slice(0, rows-1) if di == -1 else (slice(1, rows)   if di == 1 else slice(0, rows))
            c_s = slice(1, cols)   if dj == -1 else (slice(0, cols-1) if dj == 1 else slice(0, cols))
            c_n = slice(0, cols-1) if dj == -1 else (slice(1, cols)   if dj == 1 else slice(0, cols))
            src   = dem[r_s, c_s]
            nb    = dem[r_n, c_n]
            valid = np.isfinite(src) & np.isfinite(nb)
            drops[d, r_s, c_s] = np.where(valid, (src - nb) / W[d], -np.inf)

        # Step 2: flow direction = direction of max positive drop
        cb("  D8 step 2/3 — computing flow direction…")
        best_d   = np.argmax(drops, axis=0)
        max_drop = drops[
            best_d,
            np.arange(rows)[:, None],
            np.arange(cols)[None, :],
        ]
        no_recv        = (max_drop <= 0.0) | np.isnan(dem)
        best_d[no_recv] = -1

        di_arr    = np.array([d[0] for d in D8], dtype=np.int32)
        dj_arr    = np.array([d[1] for d in D8], dtype=np.int32)
        flat_n    = rows * cols
        flat_rows = np.arange(flat_n, dtype=np.int32) // cols
        flat_cols = np.arange(flat_n, dtype=np.int32) % cols
        fdir      = best_d.ravel()
        safe_d    = np.where(fdir >= 0, fdir, 0)
        recv_r    = flat_rows + di_arr[safe_d]
        recv_c    = flat_cols + dj_arr[safe_d]
        in_bounds = (recv_r >= 0) & (recv_r < rows) & (recv_c >= 0) & (recv_c < cols)
        recv_flat = np.where(
            (fdir >= 0) & in_bounds,
            recv_r * cols + recv_c,
            -1,
        ).astype(np.int64)

        # Step 3: accumulate area downstream in topological order
        cb(f"  D8 step 3/3 — accumulating {flat_n:,} cells…")
        dem_flat   = dem.ravel()
        sort_order = np.argsort(
            -np.where(np.isnan(dem_flat), -np.inf, dem_flat.astype(np.float64))
        )
        acc = np.where(np.isnan(dem_flat), 0.0, res * res).astype(np.float64)

        check_every = max(100_000, flat_n // 50)
        for step, idx in enumerate(sort_order):
            r = recv_flat[idx]
            if r >= 0:
                acc[r] += acc[idx]
            if cancel_callback is not None and step % check_every == 0:
                if cancel_callback():
                    return None

        sca = (acc / res).reshape(rows, cols).astype(np.float32)
        sca[np.isnan(dem)] = np.nan
        return sca

    def _compute_twi(
        self,
        slope_rad: np.ndarray,
        sca: np.ndarray,
    ) -> np.ndarray:
        """TWI = ln(SCA / tan(slope)).

        Slope clamped to 1e-6 rad; SCA clamped to 1.0 m²/m.
        """
        nan_mask = np.isnan(slope_rad) | np.isnan(sca)
        tan_sl   = np.tan(np.where(slope_rad < 1e-6, 1e-6, slope_rad))
        safe_sca = np.where(np.isnan(sca) | (sca < 1.0), 1.0, sca)
        with np.errstate(invalid="ignore", divide="ignore"):
            twi = np.log(safe_sca / tan_sl).astype(np.float32)
        twi[nan_mask] = np.nan
        return twi

    def _compute_spi(
        self,
        slope_rad: np.ndarray,
        sca: np.ndarray,
    ) -> np.ndarray:
        """Stream Power Index = SCA × tan(slope).

        Slope clamped to 1e-6 rad to avoid zero on flat terrain.
        """
        nan_mask = np.isnan(slope_rad) | np.isnan(sca)
        tan_sl   = np.tan(np.where(slope_rad < 1e-6, 1e-6, slope_rad))
        safe_sca = np.where(np.isnan(sca), 0.0, sca)
        spi = (safe_sca * tan_sl).astype(np.float32)
        spi[nan_mask] = np.nan
        return spi

    def _compute_ls_factor(
        self,
        slope_rad: np.ndarray,
        sca: np.ndarray,
    ) -> np.ndarray:
        """LS factor (dimensionless) — Moore & Burch (1986) / RUSLE.

        LS = (SCA / 22.13)^0.4 × (sin(slope) / 0.0896)^1.3
        SCA clamped to 1.0 m²/m; slope clamped to 1e-6 rad.
        """
        nan_mask   = np.isnan(slope_rad) | np.isnan(sca)
        safe_sca   = np.where(np.isnan(sca) | (sca < 1.0), 1.0, sca)
        safe_slope = np.where(slope_rad < 1e-6, 1e-6, slope_rad)
        with np.errstate(invalid="ignore"):
            ls = (
                (safe_sca / _LS_PLOT_LENGTH) ** _LS_M *
                (np.sin(safe_slope) / _LS_PLOT_SLOPE_SIN) ** _LS_N
            ).astype(np.float32)
        ls[nan_mask] = np.nan
        return ls
