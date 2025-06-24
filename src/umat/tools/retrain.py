import numpy as np
from cellpose.io import logger_setup
from cellpose.models import CellposeModel
from cellpose.train import train_seg
from h5py import Dataset
from h5py import File as H5File
from h5py import Group

from ..conf import RetrainConf


def get_ds(grp: Group, key: str) -> np.ndarray:
    o = grp[key]
    if not isinstance(o, Dataset):
        raise ValueError(f"expected key {key} for group {grp} to be dataset, got {o}")
    return o[:]


def run_retrain(conf: RetrainConf):
    logger_setup()

    if conf.model_path is None:
        model = CellposeModel(gpu=True, pretrained_model=str(conf.model_path))  # pyright: ignore
    else:
        model = CellposeModel(gpu=True)

    with H5File(conf.train_file, "r") as hf:
        samples = [grp for grp in hf.values() if isinstance(grp, Group) and "labels" in grp]

        images = []
        labels = []

        for sample in samples:
            if "channel: cytoplasm" not in sample.keys():
                raise ValueError(f"expected sample to have a cytoplasm channel, got: {sample}")
            if "channel: nuclear" not in sample.keys():
                raise ValueError(f"expected sample to have a nuclear channel, got: {sample}")

            cyt_chan = get_ds(sample, "channel: cytoplasm")
            nuc_chan = get_ds(sample, "channel: nuclear")
            if cyt_chan.size != nuc_chan.size:
                raise ValueError(f"expected channels to have same size, got: {cyt_chan.size} (cyt) vs {nuc_chan.size} (nuc)")

            labs = get_ds(sample, "labels")

            if cyt_chan.size != labs.size:
                raise ValueError(
                    f"expected channels and labels to have same size, got: {cyt_chan.size} (chan) vs {labs.size} (labs)"
                )

            images.append(np.stack([cyt_chan, nuc_chan, cyt_chan * 0], axis=-1))
            labels.append(labs)

    train_seg(
        model.net,
        train_data=images,
        train_labels=labels,
        weight_decay=conf.weight_decay,
        learning_rate=conf.learning_rate,
        n_epochs=conf.n_epochs,
        save_path=conf.out_path,
    )
