"""MorphometryAlgorithm -- main SwissMorph Processing algorithm.

AOI is selected via a custom widget (radio buttons):
  - Municipalities: commune names resolved via swisstopo geo.admin.ch API.
  - Map extent:     rectangle drawn on the canvas.

The AOI_CONFIG parameter stores the selection as a JSON string so that
programmatic / batch use still works without the GUI widget.
"""

import json
import os
import shutil
import tempfile
import urllib.parse
import urllib.request

from qgis.PyQt.QtCore import pyqtSignal
from qgis.PyQt.QtWidgets import (
    QHBoxLayout, QLabel, QLineEdit, QRadioButton, QVBoxLayout, QWidget,
)
from qgis.core import (
    QgsCoordinateReferenceSystem,
    QgsCoordinateTransform,
    QgsProcessingAlgorithm,
    QgsProcessingException,
    QgsProcessingLayerPostProcessorInterface,
    QgsProcessingParameterEnum,
    QgsProcessingParameterNumber,
    QgsProcessingParameterRasterDestination,
    QgsProcessingParameterString,
    QgsProject,
    QgsRasterLayer,
    QgsRectangle,
)
from processing.gui.wrappers import WidgetWrapper

from ..core.morphometry import Morphometry
from ..core.stac import StacDownloader
from .styles import (
    ls_factor_style,
    plan_curvature_style,
    profile_curvature_style,
    slope_style,
    spi_style,
    tpi_style,
    twi_style,
)

_UA = "SwissMorph-QGIS-Plugin/0.1.0"


# Custom AOI widget --------------------------------------------------------

class _AoiWidget(QWidget):
    """Radio buttons + conditional field visibility for AOI selection.

    Emits valueChanged whenever the user changes mode or content.
    getValue() / setValue() serialise the state to/from a JSON string.
    """

    valueChanged = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build()

    def _build(self):
        from qgis.gui import QgsExtentWidget

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(3)

        # Radio buttons
        rb_row = QHBoxLayout()
        self._rb_muni   = QRadioButton("Municipalities")
        self._rb_extent = QRadioButton("Map extent")
        self._rb_muni.setChecked(True)
        rb_row.addWidget(self._rb_muni)
        rb_row.addWidget(self._rb_extent)
        rb_row.addStretch()
        layout.addLayout(rb_row)

        # Municipality sub-panel (visible by default)
        self._muni_box = QWidget()
        ml = QVBoxLayout(self._muni_box)
        ml.setContentsMargins(0, 2, 0, 0)
        ml.addWidget(QLabel("Municipality names (comma or semicolon separated):"))
        self._muni_edit = QLineEdit()
        self._muni_edit.setPlaceholderText("e.g. Zermatt, Tasch, Randa")
        ml.addWidget(self._muni_edit)
        layout.addWidget(self._muni_box)

        # Extent sub-panel (hidden by default)
        self._extent_box = QWidget()
        el = QVBoxLayout(self._extent_box)
        el.setContentsMargins(0, 2, 0, 0)
        el.addWidget(QLabel("Map extent (use 'Select on canvas' to draw a rectangle):"))
        self._extent_edit = QgsExtentWidget()
        el.addWidget(self._extent_edit)
        layout.addWidget(self._extent_box)
        self._extent_box.setVisible(False)

        # Connections
        self._rb_muni.toggled.connect(self._on_mode)
        self._muni_edit.textChanged.connect(self.valueChanged)
        try:
            self._extent_edit.extentChanged.connect(self.valueChanged)
        except Exception:
            pass

    def setCanvas(self, canvas):
        if canvas is not None:
            self._extent_edit.setMapCanvas(canvas)

    def _on_mode(self, muni_selected):
        self._muni_box.setVisible(muni_selected)
        self._extent_box.setVisible(not muni_selected)
        self.valueChanged.emit()

    def getValue(self) -> str:
        """Serialise current state to JSON."""
        if self._rb_muni.isChecked():
            return json.dumps({"t": "m", "n": self._muni_edit.text().strip()})
        ext = self._extent_edit.outputExtent()
        crs = self._extent_edit.outputCrs()
        if ext and not ext.isNull() and not ext.isEmpty():
            return json.dumps({
                "t": "e",
                "b": [ext.xMinimum(), ext.yMinimum(),
                      ext.xMaximum(), ext.yMaximum()],
                "c": crs.authid() if (crs and crs.isValid()) else "EPSG:4326",
            })
        return json.dumps({"t": "e", "b": None, "c": "EPSG:4326"})

    def setValue(self, value: str):
        """Restore state from JSON string."""
        try:
            d = json.loads(value) if value else {}
        except Exception:
            d = {}
        if d.get("t") == "e":
            self._rb_extent.setChecked(True)
            box = d.get("b")
            if box and len(box) == 4:
                rect = QgsRectangle(*box)
                crs  = QgsCoordinateReferenceSystem(d.get("c", "EPSG:4326"))
                self._extent_edit.setCurrentExtent(rect, crs)
                self._extent_edit.setOutputExtentFromCurrent()
        else:
            self._rb_muni.setChecked(True)
            self._muni_edit.setText(d.get("n", ""))


class _AoiWidgetWrapper(WidgetWrapper):
    """QGIS Processing wrapper that connects AOI_CONFIG to _AoiWidget.

    Must inherit WidgetWrapper (Python) — the metadata widget_wrapper factory
    calls __init__(param, dialog, row, col), which only WidgetWrapper handles.
    """

    def createWidget(self):
        self._aoi = _AoiWidget()
        # Pass canvas so QgsExtentWidget can use "Select on canvas"
        try:
            canvas = self.dialog.mapCanvas()
            if canvas is not None:
                self._aoi.setCanvas(canvas)
        except Exception:
            pass
        return self._aoi

    def setValue(self, value):
        if hasattr(self, "_aoi") and self._aoi is not None:
            self._aoi.setValue(str(value) if value is not None else "")

    def value(self):
        if hasattr(self, "_aoi") and self._aoi is not None:
            return self._aoi.getValue()
        return ""


# Post-processor -----------------------------------------------------------

class _LayerStyler(QgsProcessingLayerPostProcessorInterface):
    """Generic post-processor: applies a style function when a raster loads.

    Stored in _instances to prevent Python GC before QGIS calls postProcessLayer.
    """

    _instances: dict = {}

    def __init__(self, style_fn):
        super().__init__()
        self._style_fn = style_fn

    def postProcessLayer(self, layer, context, feedback):
        if isinstance(layer, QgsRasterLayer):
            try:
                self._style_fn(layer)
            except Exception as exc:
                feedback.pushWarning(f"Symbology not applied: {exc}")

    @classmethod
    def create(cls, style_fn):
        inst = cls(style_fn)
        cls._instances[id(style_fn)] = inst
        return inst


# Main algorithm -----------------------------------------------------------

class MorphometryAlgorithm(QgsProcessingAlgorithm):
    """Download swissALTI3D and compute slope, curvatures, TWI, SPI, LS, TPI."""

    # Parameter / output keys
    RESOLUTION   = "RESOLUTION"
    TPI_RADIUS   = "TPI_RADIUS"
    AOI_CONFIG   = "AOI_CONFIG"
    OUTPUT_SLOPE     = "OUTPUT_SLOPE"
    OUTPUT_TWI       = "OUTPUT_TWI"
    OUTPUT_CURV      = "OUTPUT_CURV"
    OUTPUT_PROF_CURV = "OUTPUT_PROF_CURV"
    OUTPUT_SPI       = "OUTPUT_SPI"
    OUTPUT_LS        = "OUTPUT_LS"
    OUTPUT_TPI       = "OUTPUT_TPI"

    _RESOLUTIONS     = [2.0, 0.5]
    _RESOLUTION_OPTS = [
        "2 m",
        "0.5 m  (16x more data, significantly slower)",
    ]

    _SEARCH_URL = (
        "https://api3.geo.admin.ch/rest/services/api/SearchServer"
        "?searchText={q}&type=locations&origins=gg25&sr=21781&limit=1"
    )
    _LV03_TO_LV95_E = 2_000_000
    _LV03_TO_LV95_N = 1_000_000

    def initAlgorithm(self, config=None) -> None:
        self.addParameter(
            QgsProcessingParameterEnum(
                self.RESOLUTION,
                "DTM resolution",
                options=self._RESOLUTION_OPTS,
                defaultValue=0,
            )
        )

        tpi_param = QgsProcessingParameterNumber(
            self.TPI_RADIUS,
            "TPI radius in cells (1–20): 1=micro-relief, 5=meso, 10–20=macro",
            type=QgsProcessingParameterNumber.Integer,
            defaultValue=1,
            minValue=1,
            maxValue=20,
            optional=False,
        )
        tpi_param.setHelp(
            "Radius (in cells) of the neighbourhood window used to compute the "
            "Topographic Position Index (Weiss 2001). Range: 1–20.\n\n"
            "TPI = elevation of cell − mean elevation of surrounding cells. "
            "Positive values = local highs (ridges, peaks); "
            "negative values = local lows (valleys, pits).\n\n"
            "The window is always square: size = 2×radius+1. "
            "At 2 m resolution the covered diameter is (2×radius+1)×2 m:\n"
            "  1  →  3×3   (6 m)   micro-relief: ruts, furrows, small mounds\n"
            "  3  →  7×7  (14 m)   field-scale features\n"
            "  5  →  11×11 (22 m)  meso-relief: ridges, hollows, gullies\n"
            "  10 →  21×21 (42 m)  macro-relief: hills, valley floors\n"
            "  20 →  41×41 (82 m)  broad landscape units\n\n"
            "Any integer value in 1–20 is valid; computation time is the same "
            "for all radii. Default: 1."
        )
        self.addParameter(tpi_param
        )

        aoi_param = QgsProcessingParameterString(
            self.AOI_CONFIG,
            "Area of interest",
            defaultValue=json.dumps({"t": "m", "n": ""}),
            optional=False,
            multiLine=False,
        )
        aoi_param.setMetadata({"widget_wrapper": {"class": _AoiWidgetWrapper}})
        self.addParameter(aoi_param)

        self.addParameter(QgsProcessingParameterRasterDestination(
            self.OUTPUT_SLOPE, "Slope (degrees)"))
        self.addParameter(QgsProcessingParameterRasterDestination(
            self.OUTPUT_TWI, "Topographic Wetness Index (TWI)"))
        self.addParameter(QgsProcessingParameterRasterDestination(
            self.OUTPUT_CURV, "Plan curvature (1/m)"))
        self.addParameter(QgsProcessingParameterRasterDestination(
            self.OUTPUT_PROF_CURV, "Profile curvature (1/m)"))
        self.addParameter(QgsProcessingParameterRasterDestination(
            self.OUTPUT_SPI, "Stream Power Index (SPI)"))
        self.addParameter(QgsProcessingParameterRasterDestination(
            self.OUTPUT_LS, "LS factor"))
        self.addParameter(QgsProcessingParameterRasterDestination(
            self.OUTPUT_TPI, "Topographic Position Index (TPI, m)"))

    def processAlgorithm(self, parameters, context, feedback) -> dict:
        crs_lv95 = QgsCoordinateReferenceSystem("EPSG:2056")

        # 1. Resolution + TPI radius
        res_idx    = self.parameterAsEnum(parameters, self.RESOLUTION, context)
        target_res = self._RESOLUTIONS[res_idx]
        tpi_radius = self.parameterAsInt(parameters, self.TPI_RADIUS, context)
        if res_idx == 1:
            feedback.pushWarning(
                "Resolution 0.5 m selected. The raster is 16x larger than at 2 m "
                "(4x more pixels per side). Download and D8 flow accumulation "
                "will be significantly slower, especially for large AOIs."
            )

        # 2. Resolve AOI
        aoi_rect  = self._resolve_aoi(parameters, context, feedback, crs_lv95)
        bbox_lv95 = (
            aoi_rect.xMinimum(), aoi_rect.yMinimum(),
            aoi_rect.xMaximum(), aoi_rect.yMaximum(),
        )

        tmp_dir = tempfile.mkdtemp(prefix="swissmorph_")
        try:
            feedback.pushInfo(f"Resolution: {target_res} m  |  Working directory: {tmp_dir}")

            _cfg_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "config", "defaults.json",
            )
            with open(_cfg_path, encoding="utf-8") as _fh:
                _cfg = json.load(_fh)
            _cfg["stac"]["target_resolution_m"] = target_res

            downloader = StacDownloader(tmp_dir, config=_cfg)
            dtm_path = downloader.fetch(
                bbox_lv95=bbox_lv95,
                progress_callback=lambda msg: feedback.pushInfo(msg),
            )

            if feedback.isCanceled():
                return {}

            feedback.pushInfo(f"DTM ready: {dtm_path}")
            feedback.setProgress(30)

            out_slope     = self.parameterAsOutputLayer(parameters, self.OUTPUT_SLOPE,     context)
            out_twi       = self.parameterAsOutputLayer(parameters, self.OUTPUT_TWI,       context)
            out_curv      = self.parameterAsOutputLayer(parameters, self.OUTPUT_CURV,      context)
            out_prof_curv = self.parameterAsOutputLayer(parameters, self.OUTPUT_PROF_CURV, context)
            out_spi       = self.parameterAsOutputLayer(parameters, self.OUTPUT_SPI,       context)
            out_ls        = self.parameterAsOutputLayer(parameters, self.OUTPUT_LS,        context)
            out_tpi       = self.parameterAsOutputLayer(parameters, self.OUTPUT_TPI,       context)

            # Pass WBT binary path from Processing settings to core (avoids QGIS import in core/)
            try:
                from processing.core.ProcessingConfig import ProcessingConfig
                from ..core.morphometry import set_wbt_exe_hint
                wbt_path = ProcessingConfig.getSetting("WBT_EXECUTABLE") or ""
                set_wbt_exe_hint(wbt_path)
            except Exception:
                pass

            morph = Morphometry(dtm_path)

            def _morph_set_pct(pct: int) -> None:
                feedback.setProgress(30 + int(pct * 0.70))

            completed = morph.run(
                output_slope=out_slope,
                output_twi=out_twi,
                output_curvature=out_curv,
                output_profile_curvature=out_prof_curv,
                output_spi=out_spi,
                output_ls_factor=out_ls,
                output_tpi=out_tpi,
                tpi_radius=tpi_radius,
                progress_callback=lambda msg: feedback.pushInfo(msg),
                cancel_callback=feedback.isCanceled,
                progress_setter=_morph_set_pct,
            )

            if not completed or feedback.isCanceled():
                return {}

            _style_map = {
                out_slope:     slope_style,
                out_curv:      plan_curvature_style,
                out_prof_curv: profile_curvature_style,
                out_twi:       twi_style,
                out_spi:       spi_style,
                out_ls:        ls_factor_style,
                out_tpi:       tpi_style,
            }
            for _path, _fn in _style_map.items():
                if context.willLoadLayerOnCompletion(_path):
                    context.layerToLoadOnCompletionDetails(_path).setPostProcessor(
                        _LayerStyler.create(_fn)
                    )

            feedback.setProgress(100)
            feedback.pushInfo("Done.")
            return {
                self.OUTPUT_SLOPE:     out_slope,
                self.OUTPUT_TWI:       out_twi,
                self.OUTPUT_CURV:      out_curv,
                self.OUTPUT_PROF_CURV: out_prof_curv,
                self.OUTPUT_SPI:       out_spi,
                self.OUTPUT_LS:        out_ls,
                self.OUTPUT_TPI:       out_tpi,
            }

        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    # Private helpers

    def _resolve_aoi(self, parameters, context, feedback, target_crs) -> QgsRectangle:
        """Parse AOI_CONFIG JSON and return bounding box in target_crs.

        JSON schema:
          {"t": "m", "n": "Zermatt, Tasch"}          municipalities
          {"t": "e", "b": [x0,y0,x1,y1], "c": "EPSG:4326"}  map extent
        """
        raw = self.parameterAsString(parameters, self.AOI_CONFIG, context)
        try:
            aoi = json.loads(raw) if raw else {}
        except Exception:
            aoi = {}

        aoi_type = aoi.get("t", "m")

        if aoi_type == "m":
            names_str = aoi.get("n", "")
            names = [n.strip() for n in names_str.replace(";", ",").split(",") if n.strip()]
            if not names:
                raise QgsProcessingException(
                    "No municipality names provided.\n"
                    "Switch the AOI method to 'Municipalities' and type one or more "
                    "commune names (e.g. Zermatt, Tasch, Randa)."
                )
            feedback.pushInfo(f"AOI source: municipalities ({', '.join(names)})")
            xmin = ymin = float("inf")
            xmax = ymax = float("-inf")
            for name in names:
                bx0, by0, bx1, by1 = self._fetch_commune_bbox_lv95(name, feedback)
                xmin = min(xmin, bx0)
                ymin = min(ymin, by0)
                xmax = max(xmax, bx1)
                ymax = max(ymax, by1)
            aoi_rect = QgsRectangle(xmin, ymin, xmax, ymax)

        else:
            box     = aoi.get("b")
            crs_str = aoi.get("c", "EPSG:4326")
            if not box or len(box) != 4:
                raise QgsProcessingException(
                    "No map extent defined.\n"
                    "Switch the AOI method to 'Map extent' and click "
                    "'Select on canvas' to draw a rectangle."
                )
            feedback.pushInfo(f"AOI source: map extent ({crs_str})")
            ext     = QgsRectangle(*box)
            src_crs = QgsCoordinateReferenceSystem(crs_str)
            if src_crs != target_crs:
                transform = QgsCoordinateTransform(src_crs, target_crs, QgsProject.instance())
                aoi_rect  = transform.transformBoundingBox(ext)
            else:
                aoi_rect  = ext

        feedback.pushInfo(
            f"AOI in EPSG:2056 -- "
            f"xmin={aoi_rect.xMinimum():.0f} ymin={aoi_rect.yMinimum():.0f} "
            f"xmax={aoi_rect.xMaximum():.0f} ymax={aoi_rect.yMaximum():.0f}"
        )
        return aoi_rect

    def _fetch_commune_bbox_lv95(self, name: str, feedback) -> tuple:
        """Look up a municipality bounding box via geo.admin.ch SearchServer.

        Returns (xmin, ymin, xmax, ymax) in EPSG:2056.
        Raises QgsProcessingException if not found or request fails.
        """
        url = self._SEARCH_URL.format(q=urllib.parse.quote(name.strip(), safe=""))
        req = urllib.request.Request(url, headers={"User-Agent": _UA})
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except Exception as exc:
            raise QgsProcessingException(
                f"Municipality search failed for '{name}': {exc}"
            ) from exc

        results = data.get("results", [])
        if not results:
            raise QgsProcessingException(
                f"Municipality '{name}' not found. "
                "Check the spelling or use the Map extent method instead."
            )

        attrs   = results[0]["attrs"]
        label   = attrs.get("label", name).replace("<b>", "").replace("</b>", "")
        box_wkt = attrs.get("geom_st_box2d", "")
        try:
            inner = box_wkt[4:-1]
            lo, hi = inner.split(",")
            x0, y0 = [float(v) for v in lo.split()]
            x1, y1 = [float(v) for v in hi.split()]
        except Exception as exc:
            raise QgsProcessingException(
                f"Could not parse bbox for '{name}' (got: {box_wkt!r}): {exc}"
            ) from exc

        feedback.pushInfo(
            f"Matched: {label}  "
            f"(LV03 bbox: {x0:.0f},{y0:.0f} -> {x1:.0f},{y1:.0f})"
        )
        return (
            x0 + self._LV03_TO_LV95_E,
            y0 + self._LV03_TO_LV95_N,
            x1 + self._LV03_TO_LV95_E,
            y1 + self._LV03_TO_LV95_N,
        )

    # Algorithm metadata

    def name(self) -> str:
        return "terrain_morphometry"

    def displayName(self) -> str:
        return "Terrain Morphometry (swissALTI3D)"

    def group(self) -> str:
        return "Morphometry"

    def groupId(self) -> str:
        return "morphometry"

    def shortHelpString(self) -> str:
        return """
<p>Downloads <b>swissALTI3D</b> DTM tiles from the swisstopo STAC API and computes
seven morphometric layers, all in <b>EPSG:2056</b> (CH1903+/LV95).
Symbology is applied automatically when layers load into the project.</p>

<h3>Output layers</h3>

<b>Slope (degrees)</b><br>
Steepness of the terrain surface (0 = flat, 90 = vertical cliff).
Used for: erosion risk, landslide susceptibility, site suitability, habitat mapping.
<br><br>

<b>Plan curvature (1/m)</b><br>
Curvature perpendicular to the slope (horizontal).
<i>Negative / blue</i> = flow converges, moisture accumulates.
<i>Positive / red</i> = flow diverges, terrain is exposed and drier.
Used for: soil moisture patterns, geomorphological unit delineation.
<br><br>

<b>Profile curvature (1/m)</b><br>
Curvature along the slope direction (vertical).
<i>Negative / blue</i> = flow accelerates downslope (erosion risk).
<i>Positive / red</i> = flow decelerates (deposition risk).
Used for: erosion/deposition zone mapping, debris flow modelling.
<br><br>

<b>TWI - Topographic Wetness Index</b><br>
TWI = ln(SCA / tan(slope)). Tendency of each cell to accumulate water.
<i>High values</i> = wet, poorly drained, potential waterlogging.
Used for: soil moisture mapping, wetland delineation, hydrological modelling.
<br><br>

<b>SPI - Stream Power Index</b><br>
SPI = SCA x tan(slope). Proxy for the erosive power of water flow.
<i>High values</i> = high erosive energy, gully formation risk.
Used for: gully erosion susceptibility, stream channel delineation.
<br><br>

<b>LS factor (RUSLE)</b><br>
Slope-length and steepness factor of the Revised Universal Soil Loss Equation.
<i>High values</i> = high soil erosion potential.
Used for: quantitative soil loss modelling, conservation planning.
<br><br>

<b>TPI - Topographic Position Index</b><br>
Difference between a cell's elevation and the mean of its (2r+1)×(2r+1) neighbourhood
(default r=1 → 3×3, 8 neighbours; r=3 → 7×7, 48 neighbours).
<i>Positive / red</i> = ridges, hilltops.
<i>Negative / blue</i> = valleys, depressions.
<i>Near zero</i> = flat areas or mid-slopes.
A larger radius captures broader landforms (e.g. r=5 distinguishes mountain
summits from local ridgelines).
Used for: landform classification, habitat modelling, wind exposure mapping.
<br><br>

<h3>Area of interest</h3>
Select the method with the radio buttons:
<ul>
<li><b>Municipalities</b>: type commune names separated by commas
  (e.g. <i>Zermatt, Tasch, Randa</i>). Resolved via swisstopo geo.admin.ch.</li>
<li><b>Map extent</b>: click <i>Select on canvas</i> and draw a rectangle.</li>
</ul>

<h3>DTM resolution</h3>
<ul>
<li><b>2 m</b>: standard swissALTI3D resolution. Suitable for most analyses.</li>
<li><b>0.5 m</b>: 16x more data. Significantly slower. Use only for small AOIs.</li>
</ul>
"""

    def createInstance(self):
        return MorphometryAlgorithm()
