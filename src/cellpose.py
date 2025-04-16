from dataclasses import dataclass
from functools import wraps
from inspect import signature
from os import mkdir
from os.path import basename
from pathlib import Path
from typing import Annotated

import cappa
import numpy as np
from cellpose.contrib import distributed_segmentation
from cellpose.io import logger_setup
from cellpose.models import CellposeModel
from dask_cuda.local_cuda_cluster import LocalCUDACluster
from distributed import Client
from tifffile import imread
from zarr import array as zarray

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


@dataclass
class ModelFromPath:
    path: Path


@dataclass
class ModelFromType:
    type: str


@dataclass
class DistributedSegConf:
    img_fmt: Annotated[
        str,
        cappa.Arg(
            short="-i",
            help=(
                "pattern for input mosaic files, in python format string format"
                ". following patterns assumed present: 'c' (for channel) and 'z' (for z stack level)"
                ". example: 'data_dir/region_0/images/mosaic_{c}_z{z}.tif'"
            ),
        ),
    ]
    seg_pat: Annotated[str, cappa.Arg(short="-s", help="name of segmentation channel. example: 'PolyT'")]
    z_slices: Annotated[
        list[int],
        cappa.Arg(
            short="-z",
            action=cappa.ArgAction("append"),
            help="z slices to consider for segmentation. can be provided multiple times to specify multiple slices",
        ),
    ]
    model_source: Annotated[
        ModelFromPath | ModelFromType,
        cappa.Arg(
            short="-m",
            help="name of pre-trained cellpose model to use. example: 'cyto3'",
            parse=lambda a: ModelFromType(type=a),
            group=cappa.Group(name="model source", exclusive=True),
        ),
        cappa.Arg(
            short="-w",
            help="path to cellpose model weights to use",
            parse=lambda a: ModelFromPath(Path(a)),
            group=cappa.Group(name="model source", exclusive=True),
        ),
    ]
    out_path: Annotated[Path, cappa.Arg(short="-o", help="path for output masks npy file")]
    tempdir: Annotated[Path, cappa.Arg(short="-pt", help="path to temporary directory")]
    diameter: Annotated[
        int | None,
        cappa.Arg(
            short="-d",
            help="cell diameter in pixels provided to cellpose\nif left unset will be determined autonomously by cellpose",
        ),
    ] = None
    batch_size: Annotated[int, cappa.Arg(short="-b", help="batch size specified to 'Cellpose.eval'")] = (
        signature(CellposeModel.eval).parameters["batch_size"].default
    )
    chunk_x: Annotated[int, cappa.Arg(short="-cx", help="chunk block x-axis length, default")] = 256
    chunk_y: Annotated[int, cappa.Arg(short="-cy", help="chunk block y-axis length, default")] = 256
    chunk_z: Annotated[int, cappa.Arg(short="-cz", help="chunk block z-axis length, default")] = 256
    nuc_pat: Annotated[str | None, cappa.Arg(short="-n", help="name of optional nuclear channel\nexample: 'DAPI'")] = None
    cellprob_threshold: Annotated[float, cappa.Arg(short="-tc", help="cellprob_threshold specified to 'Cellpose.eval'")] = (
        signature(CellposeModel.eval).parameters["cellprob_threshold"].default
    )
    flow_threshold: Annotated[
        float | None,
        cappa.Arg(
            short="-tf", help="optional flow_threshold specified to 'Cellpose.eval', ignored if stitch_threshold is left unset"
        ),
    ] = None
    stitch_threshold: Annotated[
        float | None,
        cappa.Arg(
            short="-ts",
            help="optional stitch_threshold specified to 'Cellpose.eval', leave unset to use 'true' 3D segmentation",
        ),
    ] = None


def run_segd(conf: DistributedSegConf, cluster: LocalCUDACluster):
    seg_paths = [Path(conf.img_fmt.format(c=conf.seg_pat, z=z)) for z in conf.z_slices]
    print(f"creating segmentation channel zarr array from paths {seg_paths}", flush=True)
    seg_zarr = zarray(
        np.stack([imread(p, aszarr=False) for p in seg_paths], axis=0),
        store=conf.tempdir / "seg.zarr",
        chunks=(conf.chunk_z, conf.chunk_x, conf.chunk_y),
    )
    preprocessing_steps = []
    if conf.nuc_pat is not None:
        nuc_paths = [Path(conf.img_fmt.format(c=conf.nuc_pat, z=z)) for z in conf.z_slices]
        print(f"creating nuclear channel zarr array from paths {nuc_paths}", flush=True)
        nuc_zarr = zarray(
            np.stack([imread(p, aszarr=False) for p in nuc_paths], axis=0),
            store=conf.tempdir / "nuc.zarr",
            chunks=(conf.chunk_z, conf.chunk_x, conf.chunk_y),
        )

        def add_nuc_chan(image, crop):
            return np.stack((image, nuc_zarr[crop]), axis=-1)

        preprocessing_steps.append((add_nuc_chan, {}))

    mkdir(conf.tempdir / "cellpose_temp")

    print("running distributed_eval", flush=True)
    masks, _ = distributed_segmentation.distributed_eval(
        input_zarr=seg_zarr,
        blocksize=(conf.chunk_z, conf.chunk_x, conf.chunk_y),
        write_path=str(conf.tempdir / "out.zarr"),
        preprocessing_steps=preprocessing_steps,
        model_kwargs={
            "gpu": True,
        }
        | (
            {"model_type": conf.model_source.type}
            if isinstance(conf.model_source, ModelFromType)
            else {"pretrained_model": str(conf.model_source.path)}
        ),
        eval_kwargs={
            "batch_size": conf.batch_size,
            "channels": [0, 0] if conf.nuc_pat is None else [1, 2],
            "channel_axis": None if conf.nuc_pat is None else -1,
            "diameter": conf.diameter,
            "z_axis": 0,
            "cellprob_threshold": conf.cellprob_threshold,
        }
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

    with open(conf.out_path, "wb") as nf:
        np.save(
            nf,
            masks[:],  # pyright: ignore
        )


def run():

    conf = cappa.parse(cappa.Command(DistributedSegConf, name=basename(__file__)), completion=False)
    print(f"config: {conf}", flush=True)

    logger_setup()

    # NOTE: not using context manager due to crashes during __exit__ call

    # monkey-patch necessary since distributed_eval
    # assumes presence of a .client field in cluster
    cluster = LocalCUDACluster(
        local_directory=str(conf.tempdir / "dask_cuda_tmp"),
        shared_filesystem=True,
    )
    cluster.client = Client(cluster)  # pyright: ignore

    run_segd(conf, cluster)

    # don't fail from timeout errors on client/cluster close
    try:
        cluster.close()
    except TimeoutError:
        print("timeout error on cluster close", flush=True)
    try:
        cluster.client.close()  # pyright: ignore
    except TimeoutError:
        print("timeout error on client close", flush=True)


if __name__ == "__main__":
    run()
