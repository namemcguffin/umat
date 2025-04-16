from dataclasses import dataclass
from os.path import basename
from pathlib import Path
from random import randint
from typing import Annotated

import cappa
from h5py import File as H5File
from tifffile import imread


@dataclass
class SampleConf:
    inp_fmt: Annotated[
        str,
        cappa.Arg(
            short="-i",
            help=(
                "pattern for input mosaic files, in python format string format."
                " following patterns assumed present: 'c' (for channel)."
                " example: 'data_dir/region_0/images/mosaic_{c}_z3.tif'"
            ),
        ),
    ]
    seg_pat: Annotated[str, cappa.Arg(short="-s", help="name of segmentation channel. example: 'PolyT'")]
    amount: Annotated[int, cappa.Arg(short="-a", help="amount of samples to generate")]
    width: Annotated[int, cappa.Arg(short="-x", help="width of sample images")]
    height: Annotated[int, cappa.Arg(short="-y", help="height of sample images")]
    out_path: Annotated[Path, cappa.Arg(short="-o", help="output hdf5 file path")]
    sample_fmt: Annotated[
        str,
        cappa.Arg(
            short="-f",
            help=(
                "pattern for group name assigned to each sample."
                " following patterns assumed present: 'i' (for channel)."
                " example: 'z3 - sample: {i}'"
            ),
        ),
    ]
    nuc_pat: Annotated[str | None, cappa.Arg(short="-n", help="")] = None


def run_sampling(conf: SampleConf):
    seg_path = Path(conf.inp_fmt.format(c=conf.seg_pat))

    print(f"reading segementation image from path {seg_path}", flush=True)
    seg_arr = imread(seg_path, aszarr=False)

    # sanity check re: sampling size vs image size
    assert (
        seg_arr.shape[1] >= conf.width
    ), f"segmentation image width {seg_arr.shape[1]} is less than provided sample width {conf.width}"
    assert (
        seg_arr.shape[1] >= conf.width
    ), f"segmentation image height {seg_arr.shape[0]} is less than provided sample height {conf.height}"

    nuc_arr = None
    if conf.nuc_pat is not None:
        nuc_path = Path(conf.inp_fmt.format(c=conf.nuc_pat))
        print(f"reading nuclear image from path {nuc_path}", flush=True)
        nuc_arr = imread(nuc_path, aszarr=False)
        assert (
            seg_arr.shape == nuc_arr.shape
        ), f"segmentation image shape {seg_arr.shape[1]}x{seg_arr.shape[0]} different from nuclear image shape {nuc_arr.shape[1]}x{nuc_arr.shape[0]}"

    with H5File(conf.out_path, "a") as hf:
        for i in range(conf.amount):
            min_x = randint(0, seg_arr.shape[1] - conf.width)
            min_y = randint(0, seg_arr.shape[0] - conf.height)

            print(f"n={i}: creating group", flush=True)
            sample_grp = hf.create_group(conf.sample_fmt.format(i=i))

            print(f"n={i}: writing segmentation channel data", flush=True)
            sample_grp.create_dataset(
                "channel: segmentation", data=seg_arr[min_y : (min_y + conf.height), min_x : (min_x + conf.width)]
            )

            if nuc_arr is not None:
                print(f"n={i}: writing nuclear channel data", flush=True)
                sample_grp.create_dataset(
                    "channel: nuclear", data=nuc_arr[min_y : (min_y + conf.height), min_x : (min_x + conf.width)]
                )


def run():
    conf = cappa.parse(cappa.Command(SampleConf, name=basename(__file__)), completion=False)
    print(f"config: {conf}", flush=True)

    run_sampling(conf)


if __name__ == "__main__":
    run()
