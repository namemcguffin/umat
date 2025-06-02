from dataclasses import dataclass
from os.path import basename
from pathlib import Path
from typing import Annotated

import cappa
import numpy as np
from PIL import Image
from tifffile import imread


@dataclass
class PreviewConf:
    inp_fmt: Annotated[
        str,
        cappa.Arg(
            short="-i",
            help="pattern for input mosaic files, in python format string format."
            " following patterns assumed present: 'c' (for channel) and 'z' (for z stack level)."
            " example: 'data_dir/region_0/images/mosaic_{c}_z{z}.tif'",
        ),
    ]
    cyt_pat: Annotated[str, cappa.Arg(short="-c", help="name of cytoplasm channel\nexample: 'PolyT'")]
    nuc_pat: Annotated[str, cappa.Arg(short="-n", help="name of nuclear channel\nexample: 'DAPI'")]
    seg_masks: Annotated[Path, cappa.Arg(short="-m", help="input masks npy file path")]
    masks_z: Annotated[int, cappa.Arg(short="-z", help="z slice to consider for preview generation")]
    out_path: Annotated[Path, cappa.Arg(short="-o", help="output image showing segmentation preview path")]
    blend: Annotated[float, cappa.Arg(short="-b", help="blend value between masks and channel image")] = 0.5


def rescale_uint8(arr: np.ndarray) -> np.ndarray:
    return (((arr - np.min(arr)) / np.ptp(arr)) * 255).round().astype(np.uint8)


def run_preview(conf: PreviewConf):
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


def run():

    conf = cappa.parse(cappa.Command(PreviewConf, name=basename(__file__)), completion=False)
    print(f"config: {conf}", flush=True)

    run_preview(conf)


if __name__ == "__main__":
    run()
