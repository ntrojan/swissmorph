# SwissMorph

QGIS 3 Processing plugin for automated terrain morphometric analysis in Switzerland.

Given an area of interest, SwissMorph downloads swissALTI3D DTM tiles from the swisstopo STAC API and derives seven morphometric layers, all in EPSG:2056 (CH1903+/LV95).

## Output layers

| Layer | Unit | Method |
|---|---|---|
| Slope | degrees | Zevenbergen and Thorne (1987) |
| Plan curvature | 1/m | Zevenbergen and Thorne (1987) |
| Profile curvature | 1/m | Zevenbergen and Thorne (1987) |
| TWI | dimensionless | ln(SCA / tan(slope)) |
| SPI | m | SCA x tan(slope) |
| LS factor | dimensionless | Moore and Burch (1986) |
| TPI | m | Weiss (2001) |

## Requirements

- QGIS 3.20 or later
- Internet connection (tiles are downloaded at runtime)
- AOI must be within Switzerland

The plugin uses rasterio, numpy and pyproj, which are bundled with QGIS on all platforms. No additional installation is required.

**Optional:** Install [WhiteboxTools](https://www.whiteboxgeo.com/) for faster D8 flow accumulation on large rasters (above 4 million cells). The plugin detects it automatically if the `whitebox` Python package is installed or if the `wbt_for_qgis` QGIS plugin is configured.

## Installation

Install directly from the QGIS Plugin Manager: search for **SwissMorph**.

For manual installation, download the ZIP from the [releases page](https://github.com/ntrojan/swissmorph/releases), then use Plugin Manager > Install from ZIP.

## Usage

1. Open the Processing Toolbox (Verarbeitung > Werkzeugkiste).
2. Navigate to **SwissMorph > Terrain Morphometry (swissALTI3D)**.
3. Set the area of interest: type one or more Swiss municipality names, or draw a rectangle on the map canvas.
4. Choose the DTM resolution (2 m or 0.5 m).
5. Set the TPI radius (1 to 20 cells; see parameter help for guidance).
6. Run the algorithm. Tiles are downloaded, mosaicked and processed automatically.

Output layers are loaded into the QGIS project with automatic symbology.

## Area of interest

**Municipality mode:** type the name of one or more Swiss communes (comma-separated). Names are resolved via the swisstopo geo.admin.ch API. The bounding box of all matched communes is used.

**Map extent mode:** draw a rectangle directly on the map canvas using the "Select on canvas" button.

## Notes on performance

- At 2 m resolution, each 1 km x 1 km tile is about 500 x 500 pixels. A typical Swiss commune requires 50 to 150 tiles.
- D8 flow accumulation is the slowest step for large AOIs. WhiteboxTools (optional) can reduce this from several minutes to a few seconds.
- Only the most recent acquisition year is downloaded for each tile position, avoiding duplicate downloads from multiple survey campaigns.

## License

MIT License. See [LICENSE](LICENSE) for details.

## References

- Moore, I.D. and Burch, G.J. (1986). Physical basis of the length-slope factor in the Universal Soil Loss Equation. *Soil Science Society of America Journal*, 50(5), 1294-1298.
- Weiss, A. (2001). Topographic position and landforms analysis. Poster presentation, ESRI User Conference, San Diego, CA.
- Zevenbergen, L.W. and Thorne, C.R. (1987). Quantitative analysis of land surface topography. *Earth Surface Processes and Landforms*, 12(1), 47-56.
