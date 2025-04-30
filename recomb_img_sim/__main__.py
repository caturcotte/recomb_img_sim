#!/usr/bin/env python

import json
import numpy as np
import os

from PIL import Image

from backcross import *
from crossovers import *
from full_tracts import *
from img_classes import *
from noncrossovers import *
from read_depth import *
from read_length import *
from sequencing_errors import *
from utils import *

"""Recombination image simulation.

This module takes parameters from config.json and uses them to generate
images simulating recombination events of different types, as well as images
where no recombination has occurred.
"""


def process_img_class(
    imgs: np.ndarray, img_cls: dict, colors: dict
) -> np.ndarray:
    """Determine which simulation to run depending on the type of image."""
    match img_cls:
        case {"event": "co", "co_type": "p1" | "p2" | "reciprocal"}:
            imgs = make_cos(imgs, colors, img_cls["co_type"])
        case {
            "event": "nco",
            "parent": "p1" | "p2",
            "background": "homozygous" | "heterozygous",
        }:
            imgs = make_ncos(
                imgs, colors, img_cls["parent"], img_cls["background"]
            )
        case {"event": "homozygous", "parent": "p1" | "p2"}:
            imgs = make_homozygous(imgs, colors[img_cls["parent"]])
        case {"event": "heterozygous"}:
            imgs = make_heterozygous(imgs, colors)
        case _:
            raise ValueError(f"Invalid image class dict {img_cls}")
    return imgs


def img_sim(empty_imgs: np.ndarray, img_cls: dict, config: dict) -> np.ndarray:
    """Process all of the images in a class."""
    processed_imgs = process_img_class(empty_imgs, img_cls, config["colors"])
    imgs_with_backcross = make_backcross_reads(
        processed_imgs, config["colors"][config["backcross_parent"]]
    )
    imgs_with_errors = make_sequencing_errors(
        imgs_with_backcross, config["colors"], config["error_rate"]
    )
    imgs_shortened_reads = shorten_reads(
        imgs_with_errors,
        config["colors"]["missing_data"],
        config["read_length_mean"],
        config["read_length_stdev"],
    )
    imgs_with_reads_removed = remove_some_reads(
        imgs_shortened_reads,
        config["colors"]["missing_data"],
        config["read_depth_mean"],
        config["read_depth_stdev"],
    )
    return imgs_with_reads_removed


def main():
    """Run the simulation."""
    with open("config.json", "r") as file:
        config = json.load(file)
    output_dir = os.path.join(os.getcwd(), config["output_dir"])
    for img_cls in img_classes:
        img_cls_name = "_".join(list(img_cls.values()))
        print(f"Processing class {img_cls_name}...")
        os.makedirs(os.path.join(output_dir, img_cls_name), exist_ok=True)
        empty_imgs = make_img_array(
            config["n_images_per_class"],
            config["bin_length"],
            config["read_depth"],
            len(config["colors"]["p1"]),
        )
        final_imgs = img_sim(empty_imgs, img_cls, config)
        final_imgs = np.moveaxis(final_imgs, 1, 2)
        final_imgs = final_imgs.astype(np.uint8)
        for img in range(config["n_images_per_class"]):
            png = Image.fromarray(final_imgs[img])
            png.save(
                os.path.join(
                    output_dir, img_cls_name, f"{img_cls_name}_{img}.png"
                )
            )
        print("Done!")


if __name__ == "__main__":
    main()
