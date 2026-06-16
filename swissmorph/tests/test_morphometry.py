"""Unit tests for core/morphometry.py.

Run without QGIS:
    python -m pytest swissmorph/tests/test_morphometry.py -v
    python -m unittest discover -s swissmorph/tests -v
"""

import math
import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from swissmorph.core.morphometry import Morphometry


def _morph() -> Morphometry:
    m = Morphometry.__new__(Morphometry)
    m._dtm_path = "/dev/null"
    return m


# ── Slope ─────────────────────────────────────────────────────────────────────

class TestComputeSlope(unittest.TestCase):

    def test_flat_dem_gives_zero_slope(self):
        dem = np.zeros((9, 9), dtype=np.float32)
        slope = _morph()._compute_slope(dem, res=1.0)
        np.testing.assert_allclose(slope[2:-2, 2:-2], 0.0, atol=1e-5)

    def test_unit_eastward_slope_gives_45_degrees(self):
        rows, cols = 7, 11
        dem = np.tile(np.arange(cols - 1, -1, -1, dtype=np.float32), (rows, 1))
        slope = _morph()._compute_slope(dem, res=1.0)
        np.testing.assert_allclose(slope[2:-2, 2:-2], 45.0, atol=0.05)

    def test_slope_scales_with_resolution(self):
        rows, cols = 7, 11
        dem = np.tile(np.arange(cols - 1, -1, -1, dtype=np.float32), (rows, 1))
        s1 = _morph()._compute_slope(dem, res=1.0)[3, 5]
        s2 = _morph()._compute_slope(dem, res=2.0)[3, 5]
        self.assertAlmostEqual(s1, math.degrees(math.atan(1.0)), places=2)
        self.assertAlmostEqual(s2, math.degrees(math.atan(0.5)), places=2)

    def test_nan_cells_propagate(self):
        dem = np.ones((7, 7), dtype=np.float32)
        dem[3, 3] = np.nan
        slope = _morph()._compute_slope(dem, res=1.0)
        self.assertTrue(np.isnan(slope[3, 3]))

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
        dem = np.full((9, 9), 100.0, dtype=np.float32)
        curv = _morph()._compute_plan_curvature(dem, res=1.0)
        np.testing.assert_allclose(curv[2:-2, 2:-2], 0.0, atol=1e-5)

    def test_linear_slope_gives_zero_curvature(self):
        rows, cols = 9, 9
        dem = np.tile(np.arange(cols, dtype=np.float32), (rows, 1))
        curv = _morph()._compute_plan_curvature(dem, res=1.0)
        np.testing.assert_allclose(curv[2:-2, 2:-2], 0.0, atol=1e-5)

    def test_concave_surface_is_negative(self):
        r = np.arange(-4, 5, dtype=np.float32)
        c = np.arange(-4, 5, dtype=np.float32)
        dem = r[:, None] ** 2 + c[None, :] ** 2
        curv = _morph()._compute_plan_curvature(dem, res=1.0)
        self.assertTrue(np.all(curv[1:4, 5:8] < 0))

    def test_nan_propagates(self):
        dem = np.ones((7, 7), dtype=np.float32)
        dem[3, 3] = np.nan
        curv = _morph()._compute_plan_curvature(dem, res=1.0)
        self.assertTrue(np.isnan(curv[3, 3]))

    def test_output_shape_matches_input(self):
        dem = np.random.rand(11, 13).astype(np.float32)
        curv = _morph()._compute_plan_curvature(dem, res=1.0)
        self.assertEqual(curv.shape, dem.shape)


# ── Profile curvature ─────────────────────────────────────────────────────────

class TestComputeProfileCurvature(unittest.TestCase):

    def test_flat_dem_gives_zero(self):
        dem = np.full((9, 9), 50.0, dtype=np.float32)
        kprof = _morph()._compute_profile_curvature(dem, res=1.0)
        np.testing.assert_allclose(kprof[2:-2, 2:-2], 0.0, atol=1e-5)

    def test_linear_slope_gives_zero(self):
        """A planar slope has no curvature in any direction."""
        rows, cols = 9, 9
        dem = np.tile(np.arange(cols, dtype=np.float32), (rows, 1))
        kprof = _morph()._compute_profile_curvature(dem, res=1.0)
        np.testing.assert_allclose(kprof[2:-2, 2:-2], 0.0, atol=1e-5)

    def test_nan_propagates(self):
        dem = np.ones((7, 7), dtype=np.float32)
        dem[3, 3] = np.nan
        kprof = _morph()._compute_profile_curvature(dem, res=1.0)
        self.assertTrue(np.isnan(kprof[3, 3]))

    def test_output_shape_matches_input(self):
        dem = np.random.rand(11, 13).astype(np.float32)
        kprof = _morph()._compute_profile_curvature(dem, res=1.0)
        self.assertEqual(kprof.shape, dem.shape)

    def test_output_dtype_is_float32(self):
        dem = np.zeros((5, 5), dtype=np.float32)
        kprof = _morph()._compute_profile_curvature(dem, res=1.0)
        self.assertEqual(kprof.dtype, np.float32)

    def test_differs_from_plan_curvature_on_asymmetric_surface(self):
        """Plan and profile curvature differ on a non-radially-symmetric surface."""
        rows, cols = 9, 9
        r = np.arange(rows, dtype=np.float32)
        # z increases steeply northward, gently eastward — plan ≠ profile
        dem = r[:, None] ** 2 + 0.1 * np.arange(cols, dtype=np.float32)[None, :]
        kplan = _morph()._compute_plan_curvature(dem, res=1.0)
        kprof = _morph()._compute_profile_curvature(dem, res=1.0)
        self.assertFalse(
            np.allclose(kplan[2:-2, 2:-2], kprof[2:-2, 2:-2], atol=1e-4),
            "Plan and profile curvature must differ on an asymmetric surface",
        )


# ── TPI ───────────────────────────────────────────────────────────────────────

class TestComputeTPI(unittest.TestCase):

    def test_uniform_flat_gives_zero(self):
        dem = np.full((7, 7), 100.0, dtype=np.float32)
        tpi = _morph()._compute_tpi(dem)
        np.testing.assert_allclose(tpi[1:-1, 1:-1], 0.0, atol=1e-5)

    def test_peak_centre_is_positive(self):
        """A single raised cell has a positive TPI (higher than its neighbours)."""
        dem = np.zeros((7, 7), dtype=np.float32)
        dem[3, 3] = 10.0
        tpi = _morph()._compute_tpi(dem)
        self.assertGreater(float(tpi[3, 3]), 0.0)

    def test_valley_centre_is_negative(self):
        """A single depressed cell has a negative TPI."""
        dem = np.full((7, 7), 10.0, dtype=np.float32)
        dem[3, 3] = 0.0
        tpi = _morph()._compute_tpi(dem)
        self.assertLess(float(tpi[3, 3]), 0.0)

    def test_nan_propagates(self):
        dem = np.ones((7, 7), dtype=np.float32)
        dem[3, 3] = np.nan
        tpi = _morph()._compute_tpi(dem)
        self.assertTrue(np.isnan(tpi[3, 3]))

    def test_output_shape_matches_input(self):
        dem = np.random.rand(11, 13).astype(np.float32)
        tpi = _morph()._compute_tpi(dem)
        self.assertEqual(tpi.shape, dem.shape)

    def test_output_dtype_is_float32(self):
        dem = np.zeros((5, 5), dtype=np.float32)
        tpi = _morph()._compute_tpi(dem)
        self.assertEqual(tpi.dtype, np.float32)


# ── TWI ───────────────────────────────────────────────────────────────────────

class TestComputeTWI(unittest.TestCase):

    def test_45deg_slope_unit_sca_gives_zero(self):
        slope = np.full((3, 3), math.pi / 4, dtype=np.float32)
        sca   = np.ones((3, 3), dtype=np.float32)
        twi   = _morph()._compute_twi(slope, sca)
        np.testing.assert_allclose(twi, 0.0, atol=1e-4)

    def test_known_value(self):
        slope = np.array([[math.pi / 4]], dtype=np.float32)
        sca   = np.array([[math.e]], dtype=np.float32)
        twi   = _morph()._compute_twi(slope, sca)
        self.assertAlmostEqual(float(twi[0, 0]), 1.0, places=4)

    def test_flat_terrain_clamped_not_inf(self):
        slope = np.zeros((3, 3), dtype=np.float32)
        sca   = np.ones((3, 3), dtype=np.float32)
        twi   = _morph()._compute_twi(slope, sca)
        self.assertTrue(np.all(np.isfinite(twi)))

    def test_low_sca_clamped_to_one(self):
        slope = np.array([[math.pi / 4]], dtype=np.float32)
        sca   = np.array([[0.001]], dtype=np.float32)
        twi   = _morph()._compute_twi(slope, sca)
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


# ── SPI ───────────────────────────────────────────────────────────────────────

class TestComputeSPI(unittest.TestCase):

    def test_known_value_45deg(self):
        """SPI = SCA × tan(45°) = SCA × 1."""
        slope = np.array([[math.pi / 4]], dtype=np.float32)
        sca   = np.array([[5.0]], dtype=np.float32)
        spi   = _morph()._compute_spi(slope, sca)
        self.assertAlmostEqual(float(spi[0, 0]), 5.0, places=3)

    def test_flat_terrain_is_finite(self):
        """Zero slope is clamped → SPI must be finite (not zero/inf)."""
        slope = np.zeros((3, 3), dtype=np.float32)
        sca   = np.ones((3, 3), dtype=np.float32)
        spi   = _morph()._compute_spi(slope, sca)
        self.assertTrue(np.all(np.isfinite(spi)))

    def test_nan_in_slope_propagates(self):
        slope = np.array([[math.pi / 4, np.nan]], dtype=np.float32)
        sca   = np.array([[1.0, 1.0]], dtype=np.float32)
        spi   = _morph()._compute_spi(slope, sca)
        self.assertFalse(np.isnan(spi[0, 0]))
        self.assertTrue(np.isnan(spi[0, 1]))

    def test_nan_in_sca_propagates(self):
        slope = np.array([[math.pi / 4]], dtype=np.float32)
        sca   = np.array([[np.nan]], dtype=np.float32)
        spi   = _morph()._compute_spi(slope, sca)
        self.assertTrue(np.isnan(spi[0, 0]))

    def test_output_shape_matches_input(self):
        slope = np.full((5, 7), math.pi / 4, dtype=np.float32)
        sca   = np.ones((5, 7), dtype=np.float32)
        spi   = _morph()._compute_spi(slope, sca)
        self.assertEqual(spi.shape, slope.shape)

    def test_output_dtype_is_float32(self):
        slope = np.full((3, 3), math.pi / 4, dtype=np.float32)
        sca   = np.ones((3, 3), dtype=np.float32)
        spi   = _morph()._compute_spi(slope, sca)
        self.assertEqual(spi.dtype, np.float32)


# ── LS Factor ─────────────────────────────────────────────────────────────────

class TestComputeLSFactor(unittest.TestCase):

    def test_flat_terrain_is_finite_and_positive(self):
        """Flat terrain is clamped → LS must be finite and > 0."""
        slope = np.zeros((3, 3), dtype=np.float32)
        sca   = np.ones((3, 3), dtype=np.float32)
        ls    = _morph()._compute_ls_factor(slope, sca)
        self.assertTrue(np.all(np.isfinite(ls)))
        self.assertTrue(np.all(ls > 0))

    def test_known_value(self):
        """LS = (22.13/22.13)^0.4 × (sin(5.143°)/0.0896)^1.3 = 1.0."""
        slope_rad = np.array([[math.radians(5.143)]], dtype=np.float32)
        sca       = np.array([[22.13]], dtype=np.float32)
        ls        = _morph()._compute_ls_factor(slope_rad, sca)
        self.assertAlmostEqual(float(ls[0, 0]), 1.0, places=2)

    def test_increases_with_slope(self):
        """Higher slope → larger LS at same SCA."""
        sca      = np.array([[100.0]], dtype=np.float32)
        slope_lo = np.array([[math.radians(5.0)]], dtype=np.float32)
        slope_hi = np.array([[math.radians(20.0)]], dtype=np.float32)
        ls_lo    = _morph()._compute_ls_factor(slope_lo, sca)
        ls_hi    = _morph()._compute_ls_factor(slope_hi, sca)
        self.assertGreater(float(ls_hi[0, 0]), float(ls_lo[0, 0]))

    def test_increases_with_sca(self):
        """Greater SCA → larger LS at same slope."""
        slope   = np.array([[math.radians(10.0)]], dtype=np.float32)
        sca_lo  = np.array([[10.0]], dtype=np.float32)
        sca_hi  = np.array([[100.0]], dtype=np.float32)
        ls_lo   = _morph()._compute_ls_factor(slope, sca_lo)
        ls_hi   = _morph()._compute_ls_factor(slope, sca_hi)
        self.assertGreater(float(ls_hi[0, 0]), float(ls_lo[0, 0]))

    def test_nan_in_slope_propagates(self):
        slope = np.array([[math.pi / 4, np.nan]], dtype=np.float32)
        sca   = np.array([[1.0, 1.0]], dtype=np.float32)
        ls    = _morph()._compute_ls_factor(slope, sca)
        self.assertFalse(np.isnan(ls[0, 0]))
        self.assertTrue(np.isnan(ls[0, 1]))

    def test_nan_in_sca_propagates(self):
        slope = np.array([[math.pi / 4]], dtype=np.float32)
        sca   = np.array([[np.nan]], dtype=np.float32)
        ls    = _morph()._compute_ls_factor(slope, sca)
        self.assertTrue(np.isnan(ls[0, 0]))

    def test_output_shape_matches_input(self):
        slope = np.full((5, 7), math.pi / 4, dtype=np.float32)
        sca   = np.ones((5, 7), dtype=np.float32)
        ls    = _morph()._compute_ls_factor(slope, sca)
        self.assertEqual(ls.shape, slope.shape)

    def test_output_dtype_is_float32(self):
        slope = np.full((3, 3), math.pi / 4, dtype=np.float32)
        sca   = np.ones((3, 3), dtype=np.float32)
        ls    = _morph()._compute_ls_factor(slope, sca)
        self.assertEqual(ls.dtype, np.float32)


# ── Flow accumulation (D8) ────────────────────────────────────────────────────

class TestComputeFlowAccumulation(unittest.TestCase):
    """3×6 DEM decreasing monotonically eastward; each cell drains east."""

    RES = 1.0
    ROWS, COLS = 3, 6

    def _make_dem(self):
        return np.tile(
            np.arange(self.COLS - 1, -1, -1, dtype=np.float32), (self.ROWS, 1)
        )

    def test_source_column_sca(self):
        dem = self._make_dem()
        sca = _morph()._compute_flow_accumulation(dem, self.RES, lambda _: None)
        np.testing.assert_allclose(sca[:, 0], self.RES, rtol=1e-4)

    def test_outlet_column_sca(self):
        dem      = self._make_dem()
        sca      = _morph()._compute_flow_accumulation(dem, self.RES, lambda _: None)
        expected = float(self.COLS) * self.RES
        np.testing.assert_allclose(sca[:, -1], expected, rtol=1e-4)

    def test_accumulation_increases_eastward(self):
        dem = self._make_dem()
        sca = _morph()._compute_flow_accumulation(dem, self.RES, lambda _: None)
        for c in range(self.COLS - 1):
            self.assertTrue(np.all(sca[:, c + 1] >= sca[:, c] - 1e-4))

    def test_resolution_scales_sca(self):
        dem  = self._make_dem()
        cb   = lambda _: None
        sca1 = _morph()._compute_flow_accumulation(dem, 1.0, cb)
        sca2 = _morph()._compute_flow_accumulation(dem, 2.0, cb)
        np.testing.assert_allclose(sca2[:, -1], sca1[:, -1] * 2.0, rtol=1e-3)

    def test_nan_cells_contribute_zero_area(self):
        dem = self._make_dem()
        dem[1, 2] = np.nan
        sca = _morph()._compute_flow_accumulation(dem, self.RES, lambda _: None)
        self.assertTrue(np.isnan(sca[1, 2]))

    def test_single_cell_dem(self):
        dem = np.array([[5.0]], dtype=np.float32)
        sca = _morph()._compute_flow_accumulation(dem, 2.0, lambda _: None)
        self.assertAlmostEqual(float(sca[0, 0]), 2.0, places=4)

    def test_cancellation_returns_none(self):
        """cancel_callback returning True must abort and return None."""
        dem = self._make_dem()
        sca = _morph()._compute_flow_accumulation(
            dem, self.RES, lambda _: None, cancel_callback=lambda: True
        )
        self.assertIsNone(sca)

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
        mock_src = MagicMock()
        mock_src.crs.is_geographic = True
        mock_src.crs.to_epsg.return_value = 4326
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
        mock_src = MagicMock()
        mock_src.crs.is_geographic = False
        mock_src.res = (2.0, 2.0)
        mock_src.nodata = None
        mock_src.profile = {
            "driver": "GTiff", "dtype": "float32", "width": 3,
            "height": 3, "count": 1, "crs": "EPSG:2056", "transform": None,
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
        self.assertAlmostEqual(result[1, 1], nd,  places=5)


if __name__ == "__main__":
    unittest.main()
