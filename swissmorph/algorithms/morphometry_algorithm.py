"""MorphometryAlgorithm — main SwissMorph Processing algorithm.

Accepts an area of interest as a polygon layer and/or a map extent,
downloads swissALTI3D via the STAC API, and computes slope, plan
curvature and TWI using GDAL/numpy (no QGIS GUI imports in core/).
"""

import tempfile

from qgis.core import (
    QgsCoordinateReferenceSystem,
    QgsCoordinateTransform,
    QgsProcessingAlgorithm,
    QgsProcessingException,
    QgsProcessingParameterExtent,
    QgsProcessingParameterRasterDestination,
    QgsProcessingParameterVectorLayer,
    QgsProject,
    QgsWkbTypes,
)

from ..core.morphometry import Morphometry
from ..core.stac import StacDownloader


class MorphometryAlgorithm(QgsProcessingAlgorithm):
    """Download swissALTI3D and compute slope, TWI and plan curvature."""

    # ── Parameter / output keys ───────────────────────────────────────
    AOI_LAYER     = "AOI_LAYER"
    AOI_EXTENT    = "AOI_EXTENT"
    OUTPUT_SLOPE  = "OUTPUT_SLOPE"
    OUTPUT_TWI    = "OUTPUT_TWI"
    OUTPUT_CURV   = "OUTPUT_CURV"

    # ── Algorithm definition ──────────────────────────────────────────

    def initAlgorithm(self, config=None) -> None:
        """Declare inputs and outputs.

        AOI can be provided as a polygon layer, a map extent, or both.
        If both are given, the polygon layer takes priority.
        """
        self.addParameter(
            QgsProcessingParameterVectorLayer(
                self.AOI_LAYER,
                "Area of interest — polygon layer (takes priority)",
                types=[QgsWkbTypes.PolygonGeometry],
                optional=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterExtent(
                self.AOI_EXTENT,
                "Area of interest — map extent (used if no polygon layer)",
                optional=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterRasterDestination(
                self.OUTPUT_SLOPE,
                "Output: slope (°)",
            )
        )
        self.addParameter(
            QgsProcessingParameterRasterDestination(
                self.OUTPUT_TWI,
                "Output: Topographic Wetness Index (TWI)",
            )
        )
        self.addParameter(
            QgsProcessingParameterRasterDestination(
                self.OUTPUT_CURV,
                "Output: plan curvature (1/m)",
            )
        )

    def processAlgorithm(self, parameters, context, feedback) -> dict:
        """Run the full pipeline: resolve AOI → download DTM → compute morphometry.

        Args:
            parameters (dict): Input parameter values from the dialog.
            context (QgsProcessingContext): Processing execution context.
            feedback (QgsProcessingFeedback): Feedback object for progress and logging.

        Returns:
            dict: Mapping of output keys to output file paths.

        Raises:
            QgsProcessingException: If no AOI is provided or any step fails.
        """
        crs_lv95 = QgsCoordinateReferenceSystem("EPSG:2056")

        # ── 1. Resolve AOI ────────────────────────────────────────────
        aoi_rect = self._resolve_aoi(parameters, context, feedback, crs_lv95)

        # ── 2. Download swissALTI3D tiles for the AOI ─────────────────
        tmp_dir = tempfile.mkdtemp(prefix="swissmorph_")
        feedback.pushInfo(f"Working directory: {tmp_dir}")

        # core/ is QGIS-agnostic: extract plain float coordinates here
        bbox_lv95 = (
            aoi_rect.xMinimum(),
            aoi_rect.yMinimum(),
            aoi_rect.xMaximum(),
            aoi_rect.yMaximum(),
        )

        downloader = StacDownloader(tmp_dir)
        dtm_path = downloader.fetch(
            bbox_lv95=bbox_lv95,
            progress_callback=lambda msg: feedback.pushInfo(msg),
        )

        if feedback.isCanceled():
            return {}

        feedback.pushInfo(f"DTM ready: {dtm_path}")
        feedback.setProgress(40)

        # ── 3. Compute morphometric layers ────────────────────────────
        out_slope = self.parameterAsOutputLayer(parameters, self.OUTPUT_SLOPE, context)
        out_twi   = self.parameterAsOutputLayer(parameters, self.OUTPUT_TWI,   context)
        out_curv  = self.parameterAsOutputLayer(parameters, self.OUTPUT_CURV,  context)

        morph = Morphometry(dtm_path)
        morph.run(
            output_slope=out_slope,
            output_twi=out_twi,
            output_curvature=out_curv,
            progress_callback=lambda msg: feedback.pushInfo(msg),
        )

        if feedback.isCanceled():
            return {}

        feedback.setProgress(100)
        feedback.pushInfo("Done.")

        return {
            self.OUTPUT_SLOPE: out_slope,
            self.OUTPUT_TWI:   out_twi,
            self.OUTPUT_CURV:  out_curv,
        }

    # ── Private helpers ───────────────────────────────────────────────

    def _resolve_aoi(self, parameters, context, feedback, target_crs):
        """Determine the bounding box in *target_crs* from input parameters.

        Polygon layer takes priority over map extent if both are provided.

        Args:
            parameters (dict): Input parameter values.
            context (QgsProcessingContext): Processing context.
            feedback (QgsProcessingFeedback): Feedback for logging.
            target_crs (QgsCoordinateReferenceSystem): CRS for the output bbox.

        Returns:
            QgsRectangle: Bounding box in *target_crs*.

        Raises:
            QgsProcessingException: If neither input is provided.
        """
        layer       = self.parameterAsVectorLayer(parameters, self.AOI_LAYER,  context)
        extent_rect = self.parameterAsExtent(     parameters, self.AOI_EXTENT, context)
        extent_crs  = self.parameterAsExtentCrs(  parameters, self.AOI_EXTENT, context)

        if layer is None and (extent_rect is None or extent_rect.isEmpty()):
            raise QgsProcessingException(
                "Provide at least one AOI: a polygon layer or a map extent."
            )

        if layer is not None:
            feedback.pushInfo(f"AOI source: polygon layer '{layer.name()}'")
            source_rect = layer.extent()
            source_crs  = layer.crs()
        else:
            feedback.pushInfo("AOI source: map extent")
            source_rect = extent_rect
            source_crs  = extent_crs

        if source_crs != target_crs:
            transform   = QgsCoordinateTransform(source_crs, target_crs, QgsProject.instance())
            aoi_rect    = transform.transformBoundingBox(source_rect)
        else:
            aoi_rect    = source_rect

        feedback.pushInfo(
            f"AOI in EPSG:2056 — "
            f"xmin={aoi_rect.xMinimum():.0f} ymin={aoi_rect.yMinimum():.0f} "
            f"xmax={aoi_rect.xMaximum():.0f} ymax={aoi_rect.yMaximum():.0f}"
        )
        return aoi_rect

    # ── Algorithm metadata ────────────────────────────────────────────

    def name(self) -> str:
        return "terrain_morphometry"

    def displayName(self) -> str:
        return "Terrain Morphometry (swissALTI3D)"

    def group(self) -> str:
        return "Morphometry"

    def groupId(self) -> str:
        return "morphometry"

    def shortHelpString(self) -> str:
        return (
            "Downloads swissALTI3D 2 m DTM tiles for the given area of interest "
            "via the swisstopo STAC API, then computes:\n"
            "  • Slope (°)\n"
            "  • Topographic Wetness Index (TWI)\n"
            "  • Plan curvature (1/m)\n\n"
            "Provide a polygon layer OR a map extent (or both — polygon takes priority).\n\n"
            "All outputs are in EPSG:2056 (CH1903+ / LV95)."
        )

    def createInstance(self):
        return MorphometryAlgorithm()
