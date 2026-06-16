"""SwissMorphProvider - registers SwissMorph algorithms in the Processing Toolbox."""

from qgis.core import QgsProcessingProvider

from .algorithms.morphometry_algorithm import MorphometryAlgorithm


class SwissMorphProvider(QgsProcessingProvider):
    """Processing provider that exposes SwissMorph algorithms."""

    def loadAlgorithms(self):
        self.addAlgorithm(MorphometryAlgorithm())

    def id(self):
        return "swissmorph"

    def name(self):
        return "SwissMorph"

    def longName(self):
        return "SwissMorph - Swiss Terrain Morphometry"
