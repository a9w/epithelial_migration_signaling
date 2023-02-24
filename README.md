# epithelial_migration_signaling
This repository houses data and code used in the manuscript "Fat2 polarizes Lar and Sema5c to coordinate the motility of collectively migrating epithelial cells".

## What is and isnt here
We intended for this repository to include all the code necessary to reproduce the automated portion of the analysis in this study, as well as sample data to allow testing of that code. We also included numerical datasets that were generated more manually using ImageJ. Most of the images from which these data were generated are not included because of space constraints, but will be added to a separate repository soon and linked here. In the mean time, please reach out to the authors if you would like access.

## Structure
* Analysis of different datasets is run with either Jupyter notebooks (in the `notebooks` folder) or with python scripts in the top level of the `code` folder. Scripts should be run from within the code folder, and both will take data from and output to the `data` folder, which is subdivided by dataset. They output numerical data to the same `data` subfolder, and plots to `plots`.
* In some cases there are two datasets with the same name, except with and without the ending `_sample`. Those ending in `_sample` are the outputs of the code included here, run on the partial input datasets also included. The others are from complete input datasets.

## Note
Several of the methods included here were developed as part of a previous collaboration with Seth Donoughe ([paper here](https://doi.org/10.7554/eLife.78343)), and the associated code is available at [Fat2_polarizes_WAVE](https://github.com/a9w/Fat2_polarizes_WAVE). This includes especially the cell tracking and protrusion trait measurement methods, which Seth developed, and which are reproduced here in the interest of self-containedness.
