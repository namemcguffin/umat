from dataclasses import dataclass
from os.path import basename
from pathlib import Path
from typing import Annotated

import cappa
import numpy as np
from h5py import Dataset
from h5py import File as H5File
from h5py import Group
from roifile import ImagejRoi, roiread
from skimage.draw import polygon2mask


@dataclass
class AddLabelConf:
    hdf5_path: Annotated[Path, cappa.Arg(short="-i", help="input hdf5 file containing sampled images")]
    sample: Annotated[int, cappa.Arg(short="-s", help="sample group name")]
    lab_path: Annotated[Path, cappa.Arg(short="-l", help="input label file (can be ImageJ roi format or numpy npy format)")]


def run_addlab(conf: AddLabelConf):
    with H5File(conf.hdf5_path, "r+") as hf:
        grp = hf[conf.sample]
        if not isinstance(grp, Group):
            raise ValueError(f"entry '{conf.sample}' in provided hdf5 file {conf.hdf5_path} is not a group.")
        seg = grp["channel: cytoplasm"]
        if not isinstance(seg, Dataset):
            raise ValueError(
                f"entry 'sample: {conf.sample}/channel: cytoplasm' in provided hdf5 file {conf.hdf5_path} is not a dataset."
            )

        if conf.lab_path.suffix in (".roi", ".zip"):

            def p2m(x) -> np.ndarray:
                return polygon2mask(seg.shape, x[:, [1, 0]])

            masks = roiread(conf.lab_path)
            if isinstance(masks, ImagejRoi):
                iter = enumerate(masks.coordinates(multi=True))
            else:
                iter = enumerate(e for r in masks for e in r.coordinates(multi=True))
            labs = p2m(next(iter)[1])
            for i, e in iter:
                curr_mask = (p2m(e) * (i + 1)).astype(int)
                # handle overlapping masks by overriding with new mask where masks overlap
                labs = (labs * (curr_mask == 0)) + curr_mask
        elif conf.lab_path.suffix == ".npy":
            labs = np.load(conf.lab_path)
        else:
            raise ValueError(f"provided label file {conf.lab_path} is not of supported format")

        grp.create_dataset("labels", data=labs)


def run():
    conf = cappa.parse(cappa.Command(AddLabelConf, name=basename(__file__)), completion=False)
    print(f"config: {conf}", flush=True)

    run_addlab(conf)


if __name__ == "__main__":
    run()
