# recomb_img_sim
Generates images that simulate recombination events in genomic windows from backcrossed progeny.

## Dependencies
- `python>=3.10`
- `numpy`
- `pillow`

## Quick start
Build with `python -m pip install .` (in a venv or conda environment).
Parameters, including number of images to simulate per image class, can be configured in `config.json`.
After configuring parameters, run with `imgsim config.json`.

The default output directory is `output/`, which will be created relative to where the simulation was run. It should contain `metadata.csv`, which lists metadata for all of the images, as well as subdirectories for each image class, which contain all of the simulated images. Images are named `image_class_name_n.png`, where n is the number of the image.
