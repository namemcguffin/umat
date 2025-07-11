from pathlib import Path
from random import randint

from h5py import File as H5File
from tifffile import imread

from ..conf import SampleConf


def run(conf: SampleConf):
    cyt_path = Path(conf.inp_fmt.format(c=conf.cyt_pat))

    print(f"reading segementation image from path {cyt_path}", flush=True)
    cyt_arr = imread(cyt_path, aszarr=False)

    # sanity check re: sampling size vs image size
    assert cyt_arr.shape[1] >= conf.width, (
        f"cytoplasm image width {cyt_arr.shape[1]} is less than provided sample width {conf.width}"
    )
    assert cyt_arr.shape[1] >= conf.width, (
        f"cytoplasm image height {cyt_arr.shape[0]} is less than provided sample height {conf.height}"
    )

    nuc_path = Path(conf.inp_fmt.format(c=conf.nuc_pat))
    print(f"reading nuclear image from path {nuc_path}", flush=True)
    nuc_arr = imread(nuc_path, aszarr=False)
    assert cyt_arr.shape == nuc_arr.shape, (
        f"cytoplasm image shape {cyt_arr.shape[1]}x{cyt_arr.shape[0]} different from nuclear image shape {nuc_arr.shape[1]}x{nuc_arr.shape[0]}"
    )

    with H5File(conf.out_path, "a") as hf:
        for i in range(conf.amount):
            min_x = randint(0, cyt_arr.shape[1] - conf.width)
            min_y = randint(0, cyt_arr.shape[0] - conf.height)

            print(f"n={i}: creating group", flush=True)
            sample_grp = hf.create_group(conf.sample_fmt.format(i=i))

            print(f"n={i}: writing cytoplasm channel data", flush=True)
            sample_grp.create_dataset(
                "channel: cytoplasm", data=cyt_arr[min_y : (min_y + conf.height), min_x : (min_x + conf.width)]
            )

            print(f"n={i}: writing nuclear channel data", flush=True)
            sample_grp.create_dataset(
                "channel: nuclear", data=nuc_arr[min_y : (min_y + conf.height), min_x : (min_x + conf.width)]
            )
