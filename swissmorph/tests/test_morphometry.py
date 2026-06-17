"""Unit tests for core/morphometry.py.

Run without QGIS:
    python -m unittest discover -s swissmorph/tests -v
    # or from the repo root:
    python -m pytest swissmorph/tests/test_morphometry.py -v

All tests operate on synthetic numpy arrays — no file I/O except
TestWrite, which uses a real temp GeoTIFF via rasterio.
"""

import math
import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

# Make 'swissmorph' importable regardless of where the test is run from
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from swissmorph.core.morphometry import Morphometry


def _morph() -> Morphometry:
    """Return a Morphometry instance without a real DTM path."""
    m = Morphometry.__new__(Morphometry)
    m._dtm_path = "/dev/null"
    return m


# ── Slope ─────────────────────────────────────────────────────────────────────

class TestComputeSlope(unittest.TestCase):

    def test_flat_dem_gives_zero_slope(self):
        """A perfectly flat DEM must produce 0° slope everywhere."""
        dem = np.zeros((9, 9), dtype=np.float32)
        slope = _morph()._compute_slope(dem, res=1.0)
        np.testing.assert_allclose(slope[2:-2, 2:-2], 0.0, atol=1e-5,
                                   err_msg="Flat DEM → slope must be 0°")

    def test_unit_eastward_slope_gives_45_degrees(self):
        """DEM that drops 1 m per 1 m eastward → slope = 45° (interior cells)."""
        rows, cols = 7, 11
        # dem[r, c] = cols-1-c  →  highest at west, 0 at east, step = 1 m
        dem = np.tile(
            np.arange(cols - 1, -1, -1, dtype=np.float32), (rows, 1)
        )
        slope = _morph()._compute_slope(dem, res=1.0)
        # G = -1, H = 0 → arctan(1) = 45°
        np.testing.assert_allclose(
            slope[2:-2, 2:-2], 45.0, atol=0.05,
            err_msg="Unit eastward slope → 45°"
        )

    def test_slope_scales_with_resolution(self):
        """Same elevation change over double the pixel size → half the slope."""
        rows, cols = 7, 11
        dem = np.tile(np.arange(cols - 1, -1, -1, dtype=np.float32), (rows, 1))
        s1 = _morph()._compute_slope(dem, res=1.0)[3, 5]
        s2 = _morph()._compute_slope(dem, res=2.0)[3, 5]
        # slope1 = arctan(1/1), slope2 = arctan(1/2)
        self.assertAlmostEqual(s1, math.degrees(math.atan(1.0)), places=2)
        self.assertAlmostEqual(s2, math.degrees(math.atan(0.5)), places=2)

    def test_nan_cells_propagate(self):
        """NaN in DEM must produce NaN at the same position in slope output."""
        dem = np.ones((7, 7), dtype=np.float32)
        dem[3, 3] = np.nan
        slope = _morph()._compute_slope(dem, res=1.0)
        self.assertTrue(np.isnan(slope[3, 3]),
                        "NaN in DEM must → NaN in slope")

    def test_output_shape_matches_input(self):
        dem = np.random.rand(13, 17).astype(np.float32)
        slope = _morph()._compute_slope(dem, res=2.0)
        self.assertEqual(slope.shape, dem.shape)

    def test_output_dtype_is_float32(self):
        dem = np.zeros((5, 5), dtype=np.float32)
        slope = _morph()._compute_slope(dem, res=1.0)
        self.assertEqual(slope.dtype, np.float32)


# ── Plan curvature ────────────────────────────────────────────────────────────

class TestComputePlanCurvature(unittest.TestCase):

    def test_flat_dem_gives_zero_curvature(self):
        """Flat DEM → plan curvature = 0 everywhere."""
        dem = np.full((9, 9), 100.0, dtype=np.float32)
        curv = _morph()._compute_plan_curvature(dem, res=1.0)
        np.testing.assert_allclose(curv[2:-2, 2:-2], 0.0, atol=1e-5)

    def test_linear_slope_gives_zero_curvature(self):
        """A planar slope (no curvature) must return 0."""
        rows, cols = 9, 9
        dem = np.tile(np.arange(cols, dtype=np.float32), (rows, 1))
        curv = _morph()._compute_plan_curvature(dem, res=1.0)
        np.testing.assert_allclose(curv[2:-2, 2:-2], 0.0, atol=1e-5)

    def test_concave_surface_is_negative(self):
        """Concave bowl (converging flow) → plan curvature < 0."""
        # z = r² + c²: minimum at centre (concave, convergent flow)
        r = np.arange(-4, 5, dtype=np.float32)
        c = np.arange(-4, 5, dtype=np.float32)
        dem = r[:, None] ** 2 + c[None, :] ** 2
        curv = _morph()._compute_plan_curvature(dem, res=1.0)
        # Interior cells away from the flat centre (where gradient ≈ 0)
        self.assertTrue(
            np.all(curv[1:4, 5:8] < 0),
            "Concave bowl (z=r²+c²) → plan curvature must be negative"
        )

    def test_nan_propagates(self):
        dem = np.ones((7, 7), dtype=np.float32)
        dem[3, 3] = np.nan
        curv = _morph()._compute_plan_curvature(dem, res=1.0)
        self.assertTrue(np.isnan(curv[3, 3]))

    def test_output_shape_matches_input(self):
        dem = np.random.rand(11, 13).astype(np.float32)
        curv = _morph()._compute_plan_curvature(dem, res=1.0)
        self.assertEqual(curv.shape, dem.shape)


# ── TWI ───────────────────────────────────────────────────────────────────────

class TestComputeTWI(unittest.TestCase):

    def test_45deg_slope_unit_sca_gives_zero(self):
        """ln(SCA / tan(45°)) = ln(1/1) = 0."""
        slope = np.full((3, 3), math.pi / 4, dtype=np.float32)
        sca   = np.ones((3, 3), dtype=np.float32)
        twi   = _morph()._compute_twi(slope, sca)
        np.testing.assert_allclose(twi, 0.0, atol=1e-4)

    def test_known_value(self):
        """ln(e / tan(45°)) = ln(e) = 1."""
        slope = np.array([[math.pi / 4]], dtype=np.float32)
        sca   = np.array([[math.e]], dtype=np.float32)
        twi   = _morph()._compute_twi(slope, sca)
        self.assertAlmostEqual(float(twi[0, 0]), 1.0, places=4)

    def test_flat_terrain_clamped_not_inf(self):
        """Near-zero slope is clamped → TWI is finite (large, not inf)."""
        slope = np.zeros((3, 3), dtype=np.float32)
        sca   = np.ones((3, 3), dtype=np.float32)
        twi   = _morph()._compute_twi(slope, sca)
        self.assertTrue(np.all(np.isfinite(twi)),
                        "Flat terrain must not produce inf TWI")

    def test_low_sca_clamped_to_one(self):
        """SCA < 1 is clamped to 1 → TWI = -ln(tan(slope))."""
        slope = np.array([[math.pi / 4]], dtype=np.float32)
        sca   = np.array([[0.001]], dtype=np.float32)   # below clamp threshold
        twi   = _morph()._compute_twi(slope, sca)
        # Clamped SCA=1, slope=45° → TWI = 0
        self.assertAlmostEqual(float(twi[0, 0]), 0.0, places=4)

    def test_nan_in_slope_propagates(self):
        slope = np.array([[math.pi / 4, np.nan]], dtype=np.float32)
        sca   = np.array([[1.0, 1.0]], dtype=np.float32)
        twi   = _morph()._compute_twi(slope, sca)
        self.assertFalse(np.isnan(twi[0, 0]))
        self.assertTrue(np.isnan(twi[0, 1]))

    def test_nan_in_sca_propagates(self):
        slope = np.array([[math.pi / 4]], dtype=np.float32)
        sca   = np.array([[np.nan]], dtype=np.float32)
        twi   = _morph()._compute_twi(slope, sca)
        self.assertTrue(np.isnan(twi[0, 0]))

    def test_output_shape_matches_input(self):
        slope = np.full((5, 7), math.pi / 4, dtype=np.float32)
        sca   = np.ones((5, 7), dtype=np.float32)
        twi   = _morph()._compute_twi(slope, sca)
        self.assertEqual(twi.shape, slope.shape)


# ── Flow accumulation (D8) ────────────────────────────────────────────────────

class TestComputeFlowAccumulation(unittest.TestCase):
    """Test the D8 flow accumulation on small synthetic DEMs.

    Reference case: 3×6 DEM that decreases monotonically eastward.
    Each cell drains to its immediate east neighbour.

          col: 0  1  2  3  4  5
    row 0:     5  4  3  2  1  0
    row 1:     5  4  3  2  1  0
    row 2:     5  4  3  2  1  0

    Expected specific catchment area (SCA = acc_area / res):
      col 0:  1 × res   (only itself)
      col 1:  2 × res
      col 5:  6 × res   (full row drains through)
    """

    RES = 1.0
    ROWS, COLS = 3, 6

    def _make_dem(self):
        cols = self.COLS
        return np.tile(
            np.arange(cols - 1, -1, -1, dtype=np.float32), (self.ROWS, 1)
        )

    def test_source_column_sca(self):
        """Leftmost (highest) column → SCA = res (no upstream)."""
        dem = self._make_dem()
        sca = _morph()._compute_flow_accumulation(dem, self.RES, lambda _: None)
        np.testing.assert_allclose(
            sca[:, 0], self.RES, rtol=1e-4,
            err_msg="Source column → SCA = res (one pixel only)"
        )

    def test_outlet_column_sca(self):
        """Rightmost (lowest) column → SCA = cols × res (entire row)."""
        dem = self._make_dem()
        sca = _morph()._compute_flow_accumulation(dem, self.RES, lambda _: None)
        expected = float(self.COLS) * self.RES
        np.testing.assert_allclose(
            sca[:, -1], expected, rtol=1e-4,
            err_msg=f"Outlet column → SCA = {expected}"
        )

    def test_accumulation_increases_eastward(self):
        """SCA must increase monotonically from west to east."""
        dem = self._make_dem()
        sca = _morph()._compute_flow_accumulation(dem, self.RES, lambda _: None)
        for c in range(self.COLS - 1):
            self.assertTrue(
                np.all(sca[:, c + 1] >= sca[:, c] - 1e-4),
                f"SCA must not decrease from col {c} to {c + 1}"
            )

    def test_resolution_scales_sca(self):
        """Doubling res → SCA doubles (same drainage area, wider cells)."""
        dem  = self._make_dem()
        cb   = lambda _: None
        sca1 = _morph()._compute_flow_accumulation(dem, 1.0, cb)
        sca2 = _morph()._compute_flow_accumulation(dem, 2.0, cb)
        np.testing.assert_allclose(sca2[:, -1], sca1[:, -1] * 2.0, rtol=1e-3)

    def test_nan_cells_contribute_zero_area(self):
        """NaN cells are excluded from the drainage network."""
        dem = self._make_dem()
        dem[1, 2] = np.nan   # punch a hole in the middle row
        sca = _morph()._compute_flow_accumulation(dem, self.RES, lambda _: None)
        # The NaN cell itself
        self.assertTrue(np.isnan(sca[1, 2]),
                        "NaN DEM cell → NaN SCA at same position")

    def test_single_cell_dem(self):
        """A 1×1 DEM has no neighbours → SCA = res."""
        dem = np.array([[5.0]], dtype=np.float32)
        sca = _morph()._compute_flow_accumulation(dem, 2.0, lambda _: None)
        self.assertAlmostEqual(float(sca[0, 0]), 2.0, places=4)

    def test_output_shape_matches_input(self):
        dem = np.random.rand(5, 8).astype(np.float32)
        sca = _morph()._compute_flow_accumulation(dem, 1.0, lambda _: None)
        self.assertEqual(sca.shape, dem.shape)

    def test_output_dtype_is_float32(self):
        dem = self._make_dem()
        sca = _morph()._compute_flow_accumulation(dem, 1.0, lambda _: None)
        self.assertEqual(sca.dtype, np.float32)


# ── Geographic CRS guard ──────────────────────────────────────────────────────

class TestCRSGuard(unittest.TestCase):

    def test_geographic_crs_raises_runtime_error(self):
        """_read_dtm must raise RuntimeError when the DTM CRS is geographic."""
        mock_src = MagicMock()
        mock_src.crs.is_geographic = True
        mock_src.crs.to_epsg.return_value = 4326
        # MagicMock supports the context manager protocol by default
        mock_ctx = MagicMock()
        mock_ctx.__enter__ = MagicMock(return_value=mock_src)
        mock_ctx.__exit__ = MagicMock(return_value=False)

        with patch("swissmorph.core.morphometry.rasterio.open", return_value=mock_ctx), \
             patch("os.path.exists", return_value=True):
            m = Morphometry("/fake/wgs84.tif")
            with self.assertRaises(RuntimeError) as ctx:
                m._read_dtm()
        self.assertIn("geographic", str(ctx.exception).lower())

    def test_projected_crs_does_not_raise(self):
        """_read_dtm must succeed when the DTM CRS is projected."""
        mock_src = MagicMock()
        mock_src.crs.is_geographic = False
        mock_src.res = (2.0, 2.0)
        mock_src.nodata = None
        mock_src.profile = {
            "driver": "GTiff", "dtype": "float32", "width": 3,
            "height": 3, "count": 1, "crs": "EPSG:2056",
            "transform": None,
        }
        mock_src.read.return_value = np.zeros((3, 3), dtype=np.float32)

        mock_ctx = MagicMock()
        mock_ctx.__enter__ = MagicMock(return_value=mock_src)
        mock_ctx.__exit__ = MagicMock(return_value=False)

        with patch("swissmorph.core.morphometry.rasterio.open", return_value=mock_ctx), \
             patch("os.path.exists", return_value=True):
            m = Morphometry("/fake/lv95.tif")
            dem, res, meta = m._read_dtm()

        self.assertEqual(dem.shape, (3, 3))
        self.assertAlmostEqual(res, 2.0)


# ── Write ─────────────────────────────────────────────────────────────────────

class TestWrite(unittest.TestCase):

    def test_write_creates_valid_geotiff(self):
        """_write must produce a readable GeoTIFF with correct values."""
        try:
            import rasterio
            from rasterio.transform import from_bounds
        except ImportError:
            self.skipTest("rasterio not available in this environment")

        arr = np.array([[1.0, 2.0], [3.0, np.nan]], dtype=np.float32)
        profile = {
            "driver": "GTiff",
            "dtype": "float32",
            "width": 2,
            "height": 2,
            "count": 1,
            "crs": "EPSG:2056",
            "transform": from_bounds(600000, 200000, 600002, 200002, 2, 2),
            "nodata": -9999.0,
        }

        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "out.tif")
            _morph()._write(out, arr, profile)
            self.assertTrue(os.path.exists(out))

            with rasterio.open(out) as ds:
                result = ds.read(1)
                nd = ds.nodata

        self.assertAlmostEqual(result[0, 0], 1.0, places=5)
        self.assertAlmostEqual(result[0, 1], 2.0, places=5)
        self.assertAlmostEqual(result[1, 0], 3.0, places=5)
        self.assertAlmostEqual(result[1, 1], nd,  places=5)  # nan → nodata


if __name__ == "__main__":
    unittest.main()
