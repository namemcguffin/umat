from pathlib import Path

import numpy as np
import zarr
from PIL import Image
from tifffile import imread

from ..conf import PreviewConf


def rescale_uint8(arr: np.ndarray) -> np.ndarray:
    return (((arr - np.min(arr)) / np.ptp(arr)) * 255).round().astype(np.uint8)


def run(conf: PreviewConf):
    assert 0 <= conf.blend <= 1, ValueError("alpha blend value must be between 0 and 1")

    cyt_path = Path(conf.inp_fmt.format(c=conf.cyt_pat, z=conf.masks_z))
    nuc_path = Path(conf.inp_fmt.format(c=conf.nuc_pat, z=conf.masks_z))

    print(f"building green channel using {cyt_path}", flush=True)
    green = imread(cyt_path, aszarr=False)
    print(f"building blue channel using {nuc_path}", flush=True)
    blue = imread(nuc_path, aszarr=False)

    assert blue.dtype == green.dtype, ValueError(
        f"datatype for cytoplasm image ({green.dtype}) and nuclear image ({blue.dtype}) must be identical"
    )

    if conf.seg_masks.suffix == ".zarr":
        arr_c = zarr.open(str(conf.seg_masks), mode="r")
        assert isinstance(arr_c, zarr.Array), f"expected input file {conf.seg_masks} to contain zarr.Array, got {type(arr_c)}"
    else:
        arr_c = np.load(conf.seg_masks, mmap_mode="r")

    print(f"building red channel using masks, loaded from {conf.seg_masks} (z={conf.masks_z})", flush=True)
    red = ((arr_c[conf.masks_z, :, :] != 0) * 255).astype(np.uint8)

    print("rescaling green and blue channels, switching to uint8 dtype", flush=True)
    green = rescale_uint8(green)
    blue = rescale_uint8(blue)

    print(f"building output image, blending with alpha={conf.blend}", flush=True)
    img = Image.blend(
        Image.merge("RGB", [Image.fromarray(a) for a in (np.zeros(red.shape, np.uint8), green, blue)]),
        Image.merge("RGB", [Image.fromarray(a) for a in (red, np.zeros(red.shape, np.uint8), np.zeros(red.shape, np.uint8))]),
        conf.blend,
    )

    print(f"saving output image to {conf.out_path}", flush=True)
    img.save(conf.out_path)
