"""SwissMorph QGIS plugin — package entry point.

The top level of this package intentionally imports nothing from QGIS so
that core/ can be imported and unit-tested without a running QGIS instance.
All QGIS imports are deferred to the body of classFactory().
"""


def classFactory(iface):
    """Instantiate the plugin (called by QGIS on load).

    The import of SwissMorphPlugin is deferred so that
    'import swissmorph.core.morphometry' works outside QGIS
    (e.g. in unit tests run with plain Python).

    Args:
        iface (QgisInterface): The QGIS interface object.

    Returns:
        SwissMorphPlugin: The plugin instance.
    """
    from .plugin import SwissMorphPlugin   # deferred — needs qgis.core
    return SwissMorphPlugin(iface)
