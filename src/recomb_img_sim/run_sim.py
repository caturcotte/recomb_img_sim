import csv
import itertools
import json
import numpy as np
import os

from datetime import datetime
from PIL import Image

from .backcross import *
from .crossovers import *
from .full_tracts import *
from .img_classes import *
from .noncrossovers import *
from .read_depth import *
from .read_length import *
from .sequencing_errors import *
from .utils import *


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
    final_imgs = np.moveaxis(imgs_with_reads_removed, 1, 2).astype(np.uint8)
    return final_imgs


def save_image_get_path(imgs, img_cls_name, img, out_dir):
    """Save image as file and return file name."""
    png = Image.fromarray(imgs[img])
    out_file = os.path.join(
        out_dir,
        img_cls_name,
        f"{img_cls_name}_{img}.png",
    )
    png.save(out_file)
    return out_file


def save_images_and_metadata(
    all_imgs: dict, img_cls_names: list, n_imgs: int, out_dir: str
):
    """Save images and return their metadata."""
    for img_cls, img_number in itertools.product(img_cls_names, range(n_imgs)):
        out_file = save_image_get_path(
            all_imgs[img_cls], img_cls, img_number, out_dir
        )
        yield [img_cls, img_number, out_file]


def run_img_sim(config_file: str):
    """Run the simulation."""
    print("RECOMBINATION IMAGE SIMULATION")
    with open(config_file, "r") as file:
        config = json.load(file)
    print_parameters(config)
    output_dir = os.path.join(os.getcwd(), config["output_dir"])
    img_cls_names = ["_".join(list(i.values())) for i in img_classes]
    img_cls_imgs = {}
    for img_cls, img_cls_name in zip(img_classes, img_cls_names):
        tstamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{tstamp}] Processing class {img_cls_name}...")
        os.makedirs(os.path.join(output_dir, img_cls_name), exist_ok=True)
        empty_imgs = make_img_array(
            config["n_images_per_class"],
            config["bin_length"],
            config["read_depth"],
            len(config["colors"]["p1"]),
        )
        final_imgs = img_sim(empty_imgs, img_cls, config)
        img_cls_imgs[img_cls_name] = final_imgs
    tstamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{tstamp}] Saving images and writing metadata...")
    with open(os.path.join(output_dir, "metadata.csv"), "w") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["img_class", "img", "path"])
        img_and_metadata_generator = save_images_and_metadata(
            img_cls_imgs,
            img_cls_names,
            config["n_images_per_class"],
            output_dir,
        )
        for line in img_and_metadata_generator:
            writer.writerow(line)
    tstamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{tstamp}] Done!")
