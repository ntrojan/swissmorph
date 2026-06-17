"""SwissMorph plugin entry point.

Registers the Processing provider on load and removes it on unload.
No GUI elements are created here — the Processing framework handles the dialog.
"""

from qgis.core import QgsApplication
from .provider import SwissMorphProvider


class SwissMorphPlugin:
    """QGIS plugin lifecycle manager for SwissMorph."""

    def __init__(self, iface) -> None:
        """
        Args:
            iface (QgisInterface): The QGIS interface object (kept for future use).
        """
        self.iface = iface
        self._provider = None

    # ── QGIS lifecycle hooks ──────────────────────────────────────────

    def initProcessing(self) -> None:
        """Register the SwissMorph Processing provider."""
        self._provider = SwissMorphProvider()
        QgsApplication.processingRegistry().addProvider(self._provider)

    def initGui(self) -> None:
        """Called by QGIS when the plugin is loaded into the GUI."""
        self.initProcessing()

    def unload(self) -> None:
        """Called by QGIS when the plugin is unloaded. Cleans up the provider."""
        if self._provider is not None:
            QgsApplication.processingRegistry().removeProvider(self._provider)
            self._provider = None
