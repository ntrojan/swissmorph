"""SwissMorph Processing provider.

Registers all algorithms under the "SwissMorph" group in the
Processing Toolbox.
"""

import os

from qgis.PyQt.QtGui import QIcon
from qgis.core import QgsProcessingProvider
from .algorithms.morphometry_algorithm import MorphometryAlgorithm

_ICON_PATH = os.path.join(os.path.dirname(__file__), "resources", "icon.png")


class SwissMorphProvider(QgsProcessingProvider):
    """Processing provider for SwissMorph algorithms."""

    def loadAlgorithms(self) -> None:
        """Register all algorithms exposed by this provider."""
        self.addAlgorithm(MorphometryAlgorithm())

    def icon(self) -> QIcon:
        """Icon shown next to the provider group in the Toolbox.

        Returns:
            QIcon: The SwissMorph icon.
        """
        return QIcon(_ICON_PATH)

    def id(self) -> str:
        """Unique provider identifier (lowercase, no spaces).

        Returns:
            str: Provider ID.
        """
        return "swissmorph"

    def name(self) -> str:
        """Human-readable provider name shown in the Toolbox.

        Returns:
            str: Display name.
        """
        return "SwissMorph"

    def longName(self) -> str:
        """Extended provider name.

        Returns:
            str: Long display name.
        """
        return "SwissMorph — Swiss Terrain Morphometry"
