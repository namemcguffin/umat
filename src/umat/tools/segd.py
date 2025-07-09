from functools import wraps
from gc import collect
from os import mkdir
from pathlib import Path
from shutil import copytree

import numpy as np
from cellpose.contrib import distributed_segmentation
from cellpose.io import logger_setup
from dask_cuda.local_cuda_cluster import LocalCUDACluster
from distributed import Client
from tifffile import imread
from zarr import Array as ZArray
from zarr import array as zarray

from ..conf import DistributedSegConf

# below needed since get_block_crops assumes that overlap is always
# an int but it will be a float if diameter is set to a float
_gbc = distributed_segmentation.get_block_crops


@wraps(_gbc)
def wgbc(*args, **kwargs):
    if "overlap" in kwargs:
        kwargs["overlap"] = round(kwargs["overlap"])
    else:
        args = list(args)
        args[2] = round(args[2])
    return _gbc(*args, **kwargs)


distributed_segmentation.get_block_crops = wgbc


def run(conf: DistributedSegConf):

    logger_setup()

    # NOTE: not using context manager due to crashes during __exit__ call

    # monkey-patch necessary since distributed_eval
    # assumes presence of a .client field in cluster
    cluster = LocalCUDACluster(
        local_directory=str(conf.tempdir / "dask_cuda_tmp"),
        shared_filesystem=True,
    )
    cluster.client = Client(cluster)  # pyright: ignore

    cyt_paths = [Path(conf.img_fmt.format(c=conf.cyt_pat, z=z)) for z in conf.z_slices]
    print(f"creating cytoplasm channel zarr array from paths {cyt_paths}", flush=True)
    cyt_zarr = zarray(
        np.stack([imread(p, aszarr=False) for p in cyt_paths], axis=0),
        store=conf.tempdir / "seg.zarr",
        chunks=(conf.chunk_z, conf.chunk_x, conf.chunk_y),
    )

    nuc_paths = [Path(conf.img_fmt.format(c=conf.nuc_pat, z=z)) for z in conf.z_slices]
    print(f"creating nuclear channel zarr array from paths {nuc_paths}", flush=True)
    nuc_zarr = zarray(
        np.stack([imread(p, aszarr=False) for p in nuc_paths], axis=0),
        store=conf.tempdir / "nuc.zarr",
        chunks=(conf.chunk_z, conf.chunk_x, conf.chunk_y),
    )

    mkdir(conf.tempdir / "cellpose_temp")

    print("running distributed_eval", flush=True)
    masks, _ = distributed_segmentation.distributed_eval(
        input_zarr=cyt_zarr,
        blocksize=(conf.chunk_z, conf.chunk_x, conf.chunk_y),
        write_path=str(conf.tempdir / "out.zarr"),
        # insert nuclear channel during preprocessing step as documented
        # here: https://cellpose.readthedocs.io/en/latest/distributed.html
        preprocessing_steps=[
            (
                lambda image, crop: np.stack(
                    (
                        image,
                        nuc_zarr[crop],  # noqa: F821 - ruff seems to not handle lambda capture properly (?)
                        image * 0,
                    ),
                    axis=-1,
                ),
                {},
            )
        ],
        model_kwargs={
            "gpu": True,
        }
        | ({"pretrained_model": str(conf.model_path)} if conf.model_path is not None else {}),
        eval_kwargs={
            "batch_size": conf.batch_size,
            "channel_axis": None if conf.nuc_pat is None else -1,
            "z_axis": 0,
            "cellprob_threshold": conf.cellprob_threshold,
        }
        | ({"diameter": conf.diameter} if conf.diameter is not None else {})
        | (
            {
                "do_3D": True,
            }
            if conf.stitch_threshold is None
            else {
                "stitch_threshold": conf.stitch_threshold,
                "flow_threshold": conf.flow_threshold,
            }
        ),
        cluster=cluster,
        temporary_directory=str(conf.tempdir / "cellpose_temp"),
    )

    # sanity check to make sure masks is of right type
    assert isinstance(masks, ZArray), f"expected masks to be zarr.Array, got {type(masks)}"

    # attempt to clear up memory space
    # before loading masks into memory
    del cyt_zarr
    del cyt_paths
    del nuc_zarr
    del nuc_paths
    collect()

    if conf.out_path.suffix == ".zarr":
        copytree(conf.tempdir / "out.zarr", conf.out_path)
    else:
        with open(conf.out_path, "wb") as nf:
            np.save(nf, masks[:])

    # don't fail from timeout errors on client/cluster close
    try:
        cluster.close()
    except TimeoutError:
        print("timeout error on cluster close", flush=True)
    try:
        cluster.client.close()  # pyright: ignore
    except TimeoutError:
        print("timeout error on client close", flush=True)
