import numpy as np
import pandas as pd
import zarr
from skimage.measure import regionprops_table
from tifffile import imread

from ..conf import SignalsConf


def run(conf: SignalsConf):
    print(f"loading masks from {conf.masks_path}")
    if conf.masks_path.suffix == ".zarr":
        arr = zarr.open(str(conf.masks_path), mode="r")
        assert isinstance(arr, zarr.Array), f"expected input file {conf.masks_path} to contain zarr.Array, got {type(arr)}"
        masks = arr[slice(None) if conf.z_subset is None else sorted(conf.z_subset), :, :]  # noqa: E731
    else:
        arr = np.load(conf.masks_path, mmap_mode="r")
        masks = arr[slice(None) if conf.z_subset is None else sorted(conf.z_subset), :, :]  # noqa: E731

    imgs = []
    for c in conf.channels:
        chan = []
        for z in range(masks.shape[0]) if conf.z_subset is None else sorted(conf.z_subset):
            path = conf.inp_fmt.format(z=z, c=c)
            print(f"z={z}, c={c}: loading image from {path}", flush=True)
            chan.append(imread(path))
        imgs.append(np.stack(chan, axis=0))
    imgs = np.stack(imgs, axis=-1)

    assert masks.shape[:3] == imgs.shape[:3], (
        f"expected first 3 dimensions of masks and images to be the same, got: {masks.shape[:3]} (masks), {imgs.shape[:3]} (images)"
    )

    print("determining region properties", flush=True)
    df = pd.DataFrame(regionprops_table(masks, imgs, properties=["label", *conf.props]))

    for idx, chan in enumerate(conf.channels):
        df.rename(columns={col: col.replace(f"-{idx}", f"-{chan}") for col in df.columns}, inplace=True)

    print(f"saving signals table to {conf.out_path}", flush=True)
    df.to_csv(conf.out_path, sep="\t", index=False)
