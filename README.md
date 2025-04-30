# recomb_img_sim
Generates images that simulate recombination events in genomic windows from backcrossed progeny.

## Dependencies
- `python>=3.10`
- `numpy`
- `pillow`
## Quick start
Build with `python -m pip install .` (in a venv or conda environment).
Parameters, including number of images to simulate per image class, can be configured in `config.json`.
After configuring parameters, run with `python recomb_img_sim`.

The default output directory is `output/` in the location in which the simulation was run. It should contain subdirectories for each image class, which contain all of the simulated images.
