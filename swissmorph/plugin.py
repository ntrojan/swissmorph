"""SwissMorph QGIS plugin - lifecycle management.

No GUI elements are created here - the Processing framework handles the dialog.
"""

from qgis.core import QgsApplication

from .provider import SwissMorphProvider


class SwissMorphPlugin:
    """Plugin entry point: registers the Processing provider on load."""

    def __init__(self, iface):
        self.iface = iface
        self._provider = None

    # QGIS lifecycle hooks

    def initGui(self):
        self._provider = SwissMorphProvider()
        QgsApplication.processingRegistry().addProvider(self._provider)

    def unload(self):
        QgsApplication.processingRegistry().removeProvider(self._provider)
