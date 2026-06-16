"""Raster symbology for SwissMorph output layers.

Each function receives a QgsRasterLayer and applies a QgsSingleBandPseudoColorRenderer.
They are invoked by _LayerStyler in morphometry_algorithm.py as QGIS post-processors,
called automatically when each output is loaded into the project.
"""

from qgis.PyQt.QtGui import QColor
from qgis.core import (
    QgsColorRampShader,
    QgsRasterBandStats,
    QgsRasterShader,
    QgsSingleBandPseudoColorRenderer,
)


# Internal helpers

def _pseudo_renderer(layer, stops, band=1):
    """Build a QgsSingleBandPseudoColorRenderer from (value, '#rrggbb') stop list."""
    items = [
        QgsColorRampShader.ColorRampItem(v, QColor(c), f"{v:.3g}")
        for v, c in stops
    ]
    ramp = QgsColorRampShader()
    ramp.setColorRampType(QgsColorRampShader.Interpolated)
    ramp.setColorRampItemList(items)

    shader = QgsRasterShader()
    shader.setRasterShaderFunction(ramp)

    renderer = QgsSingleBandPseudoColorRenderer(layer.dataProvider(), band, shader)
    renderer.setClassificationMin(stops[0][0])
    renderer.setClassificationMax(stops[-1][0])
    return renderer


def _stretch(layer, band=1, n_sd=2.0):
    """Return (lo, mean, hi) = mean ± n·σ computed from actual pixel data.

    nodata cells are excluded by QGIS automatically. σ is clamped to 1e-9
    to guard against flat/invalid rasters.
    """
    stats = layer.dataProvider().bandStatistics(band, QgsRasterBandStats.All)
    sd    = max(stats.stdDev, 1e-9)
    return stats.mean - n_sd * sd, stats.mean, stats.mean + n_sd * sd


def _sym_half(layer, band=1, n_sd=2.0):
    """Return a symmetric half-range (n·σ) centred on 0 for diverging ramps."""
    stats = layer.dataProvider().bandStatistics(band, QgsRasterBandStats.All)
    return max(n_sd * stats.stdDev, 1e-9)


# Per-output style functions

def slope_style(layer):
    """Fixed 0–50° range: dark green → yellow → orange → dark red."""
    stops = [
        (0,  "#1a9850"),   # flat         → dark green
        (10, "#91cf60"),   # gentle       → light green
        (20, "#fee08b"),   # moderate     → yellow
        (30, "#fc8d59"),   # steep        → orange
        (40, "#d73027"),   # very steep   → red
        (50, "#7f0000"),   # cliff/extreme → dark red
    ]
    layer.setRenderer(_pseudo_renderer(layer, stops))
    layer.triggerRepaint()


def plan_curvature_style(layer):
    """Diverging blue → white → red centred on 0, symmetric ±2 SD.

    Blue  = concave / convergent flow.
    Red   = convex  / divergent flow.
    """
    h = _sym_half(layer)
    stops = [(-h, "#2166ac"), (0, "#f7f7f7"), (+h, "#d6604d")]
    layer.setRenderer(_pseudo_renderer(layer, stops))
    layer.triggerRepaint()


def profile_curvature_style(layer):
    """Diverging blue → white → red centred on 0, symmetric ±2 SD.

    Blue  = concave downslope (flow accelerates).
    Red   = convex  downslope (flow decelerates).
    """
    h = _sym_half(layer)
    stops = [(-h, "#4575b4"), (0, "#f7f7f7"), (+h, "#d73027")]
    layer.setRenderer(_pseudo_renderer(layer, stops))
    layer.triggerRepaint()


def tpi_style(layer):
    """Diverging blue (valley) → pale yellow (flat) → red (ridge), ±2 SD."""
    h = _sym_half(layer)
    stops = [(-h, "#4575b4"), (0, "#ffffbf"), (+h, "#d73027")]
    layer.setRenderer(_pseudo_renderer(layer, stops))
    layer.triggerRepaint()


def twi_style(layer):
    """Pale yellow (dry) → teal → dark blue (wet), mean ±2 SD."""
    lo, mid, hi = _stretch(layer)
    lo = max(lo, 0)   # TWI is always positive
    stops = [(lo, "#ffffcc"), (mid, "#41b6c4"), (hi, "#0c2c84")]
    layer.setRenderer(_pseudo_renderer(layer, stops))
    layer.triggerRepaint()


def spi_style(layer):
    """White → green → dark green, mean ±2 SD (clamped to ≥ 0)."""
    lo, mid, hi = _stretch(layer)
    lo = max(lo, 0)
    stops = [(lo, "#ffffff"), (mid, "#2ca25f"), (hi, "#005824")]
    layer.setRenderer(_pseudo_renderer(layer, stops))
    layer.triggerRepaint()


def ls_factor_style(layer):
    """Pale yellow → orange → dark red, 0 to mean+2 SD."""
    _, _, hi = _stretch(layer)
    hi = max(hi, 0.1)
    stops = [(0, "#ffffb2"), (hi / 2, "#fd8d3c"), (hi, "#bd0026")]
    layer.setRenderer(_pseudo_renderer(layer, stops))
    layer.triggerRepaint()
