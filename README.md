# epithelial_migration_signaling
This repository houses data and code used in the manuscript "Fat2 polarizes Lar and Sema5c to coordinate the motility of collectively migrating epithelial cells".

## What is and isnt here
We intended for this repository to include all the code necessary to reproduce the automated portion of the analysis in this study, as well as sample data to allow testing of that code. We also included numerical datasets that were generated more manually using ImageJ. Most of the images from which these data were generated are not included because of space constraints, but will be added to a separate repository soon and linked here. In the mean time, please reach out to the authors if you would like access.

## Structure
* Analysis of different datasets is run with either Jupyter notebooks (in the `notebooks` folder) or with python scripts in the top level of the `code` folder. Scripts should be run from within the code folder, and both will take data from and output to the `data` folder, which is subdivided by dataset. They output numerical data to the same `data` subfolder, and plots to `plots`.
* In some cases there are two datasets with the same name, except with and without the ending `_sample`. Those ending in `_sample` are the outputs of the code included here, run on the partial input datasets also included. The others are from complete input datasets.

## Note
Several of the methods included here were developed as part of a previous collaboration with Seth Donoughe ([paper here](https://doi.org/10.7554/eLife.78343)), and the associated code is available at [Fat2_polarizes_WAVE](https://github.com/a9w/Fat2_polarizes_WAVE). This includes especially the cell tracking and protrusion trait measurement methods, which Seth developed, and which are reproduced here in the interest of self-containedness.



## License
All creative content is licensed under the [Creative Commons CC-BY 4.0](https://creativecommons.org/licenses/by/4.0/) license.

All software is distributed under the MIT license:

```
Copyright (c) 2023 The Authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
