# Naming of solar panels

![jupyter](https://img.shields.io/github/languages/top/AlfonsoLRz/SolarPanelManager) 
![license](https://img.shields.io/badge/license-MIT-blue.svg)

Naming of solar panels according to their spatial distribution and clustering in panels. This repository requires a previous tool for automatically extracting quads depicting solar panels. Hence, the input of this program is a `.shp` file.

## Dependencies

Dependencies are defined in `requirements.txt`. These include `matplotlib`, `numpy`, `seaborn`, `Fiona`, `pandas` and `jupyter`.

## Processing

The workflow mainly depends on three parameters that can be found in `panel_naming.ipynb`, which are the following:

1. **Filename**: path of a `.shp` file that includes the geometry of solar panels. 

2. **Epsilon distance** in DBSCAN: maximum distance between two points.

3. **Max distance between cells in the same row**. It must be configured according to how well were cells obtained. Normally, a value below 0.5 meters works fine.

The workflow consists of two main steps:

1. **Clustering and ordering of arrays**. Arrays are the nomenclature used to refer to isolated groups of solar panels. These are first extracted with DBSCAN and then sorted in X and Y.

<p align="center">
    <img src="assets/array_map.png"/></br>
    <em>Ordered identification of arrays.</em>
</p>

2. **Clustering and ordering of cells**. The same as the previous step, but applied to panels in arrays instead of simply arrays.

<p align="center">
    <img src="assets/panel_map.png"/></br>
    <em>Ordered identification of arrays and panels.</em>
</p>

## Autofill

The script `identify_missing_panels.ipynb` can detect missing panels from a previously generated `.shp` file. From the previous step, the number of columns and rows is known for each array and thus the solar plantation can be filled with those panels that were not detected in the algorithm of your preference.

The only parameter that can be configured during this process is the threshold distance considered to decide whether the space between two consecutive panels is too much or not (`threshold_distance`). Note that this factor will be multiplied by the average panel width, and therefore, can be specified in terms of %.

<p align="center">
    <img src="assets/missing_panels.png"/></br>
    <em>Initial panel detection with missing geometry.</em>
</p>

<p align="center">
    <img src="assets/identified_panels.png"/></br>
    <em>Fixed geometry of a solar plantation.</em>
</p>

## TODO

- Make it robust to manage arrays that have different directions. Currently, a global direction is averaged and every array is considered to expand in such a direction.
