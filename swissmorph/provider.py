"""SwissMorph Processing provider.

Registers all algorithms under the "SwissMorph" group in the
Processing Toolbox.
"""

from qgis.core import QgsProcessingProvider
from .algorithms.morphometry_algorithm import MorphometryAlgorithm


class SwissMorphProvider(QgsProcessingProvider):
    """Processing provider for SwissMorph algorithms."""

    def loadAlgorithms(self) -> None:
        """Register all algorithms exposed by this provider."""
        self.addAlgorithm(MorphometryAlgorithm())

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
