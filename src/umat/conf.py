from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated

import cappa


@cappa.command(name="addlab")
@dataclass
class AddLabelConf:
    hdf5_path: Annotated[Path, cappa.Arg(short="-i", help="input hdf5 file containing sampled images")]
    sample: Annotated[int, cappa.Arg(short="-s", help="sample group name")]
    lab_path: Annotated[Path, cappa.Arg(short="-l", help="input label file (can be ImageJ roi format or numpy npy format)")]


@cappa.command(name="assign")
@dataclass
class AssignConf:
    b_paths: Annotated[
        list[Path],
        cappa.Arg(
            short="-i",
            action=cappa.ArgAction("append"),
            help="input feather file(s) containing cell geometries. can be provided multiple times, all tables will be concatenated together before processing.",
        ),
    ]
    ad_path: Annotated[Path, cappa.Arg(short="-a", help="output anndata h5ad file path")]
    ft_path: Annotated[
        Path, cappa.Arg(short="-f", help="output feather file path, containing updated transcript cell assignments")
    ]
    dt_path: Annotated[Path, cappa.Arg(short="-d", help="input detected transcripts CSV file path")]


@cappa.command(name="boundary")
@dataclass
class BoundaryConf:
    inp_path: Annotated[Path, cappa.Arg(short="-i", help="input masks file path (npy or zarr)")]
    out_path: Annotated[
        Path, cappa.Arg(short="-o", help="output feather file path containing cell geometries (geopandas format)")
    ]
    mp_path: Annotated[Path, cappa.Arg(short="-m", help="mosaic micron to mosaic pixel transform file path")]
    z_subset: Annotated[
        list[int] | None,
        cappa.Arg(
            short="-z",
            action=cappa.ArgAction("append"),
            help="z slice(s) to consider from masks file, can be provided multiple times to specify multiple slices (will be processed sequentially)",
        ),
    ] = None
    ncpus: Annotated[int, cappa.Arg(short="-j", help="amount of CPU cores to use")] = 1


@cappa.command(name="signals")
@dataclass
class SignalsConf:
    inp_fmt: Annotated[
        str,
        cappa.Arg(
            short="-i",
            help="pattern for input mosaic files, in python format string format."
            " following patterns assumed present: 'c' (for channel) and 'z' (for z stack level)."
            " example: 'data_dir/region_0/images/mosaic_{c}_z{z}.tif'",
        ),
    ]
    channels: Annotated[
        list[str],
        cappa.Arg(
            short="-c", action=cappa.ArgAction("append"), help="name of channels to load and compute signal statistics from"
        ),
    ]
    masks_path: Annotated[Path, cappa.Arg(short="-m", help="masks file path (npy or zarr)")]
    out_path: Annotated[Path, cappa.Arg(short="-o", help="output TSV file path containing aggreggated cell stats")]
    props: Annotated[
        list[str],
        cappa.Arg(
            short="-p",
            action=cappa.ArgAction("append"),
            help="name of properties to calculate (see `skimage.measure.regionprops` for options)",
        ),
    ] = field(default_factory=lambda: ["area", "intensity_mean"])
    z_subset: Annotated[
        list[int] | None,
        cappa.Arg(
            short="-z",
            action=cappa.ArgAction("append"),
            help="z slice(s) to consider, other slices will be ignored",
        ),
    ] = None


@cappa.command(name="fromproseg")
@dataclass
class FromProsegConf:
    geojson_path: Annotated[
        Path,
        cappa.Arg(short="-i", help="path to proseg generated layer-resolved geojson output file"),
    ]
    x_shape: Annotated[int, cappa.Arg(short="-x", help="width of output masks, should match mosaic image width in pixels")]
    y_shape: Annotated[int, cappa.Arg(short="-y", help="height of output masks, should match mosaic image height in pixels")]
    mp_path: Annotated[Path, cappa.Arg(short="-m", help="mosaic micron to mosaic pixel transform file path")]
    out_path: Annotated[Path, cappa.Arg(short="-o", help="path for output masks file (npy or zarr)")]
    z_slice: Annotated[int | None, cappa.Arg(short="-z", help="specify just one z-slice to run mask generation on")] = None


@cappa.command(name="preview")
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
    seg_masks: Annotated[Path, cappa.Arg(short="-m", help="input masks file path (npy or zarr)")]
    masks_z: Annotated[int, cappa.Arg(short="-z", help="z slice to consider for preview generation")]
    out_path: Annotated[Path, cappa.Arg(short="-o", help="output image showing segmentation preview path")]
    blend: Annotated[float, cappa.Arg(short="-b", help="blend value between masks and channel image")] = 0.5


@cappa.command(name="retrain")
@dataclass
class RetrainConf:
    train_file: Annotated[Path, cappa.Arg(short="-i", help="path to input training file, generated by `umat addlab`")]
    out_path: Annotated[Path, cappa.Arg(short="-o", help="path to re-trained cellpose output weights file")]
    weight_decay: Annotated[float, cappa.Arg(short="-d", help="weight decay for optimizer during training")] = 0.1
    learning_rate: Annotated[float, cappa.Arg(short="-l", help="learning rate for training")] = 0.00001
    n_epochs: Annotated[int, cappa.Arg(short="-n", help="number of epochs for training")] = 300
    model_path: Annotated[
        Path | None,
        cappa.Arg(
            short="-w",
            help="optional path to cellpose model weights to use (uses cpsam if unset)",
        ),
    ] = None


@cappa.command(name="sample")
@dataclass
class SampleConf:
    inp_fmt: Annotated[
        str,
        cappa.Arg(
            short="-i",
            help=(
                "pattern for input mosaic files, in python format string format."
                " following patterns assumed present: 'c' (for channel)."
                " example: 'data_dir/region_0/images/mosaic_{c}_z3.tif'."
            ),
        ),
    ]
    cyt_pat: Annotated[str, cappa.Arg(short="-c", help="name of cytoplasm channel. example: 'PolyT'")]
    nuc_pat: Annotated[str, cappa.Arg(short="-n", help="name of nuclear channel. example: 'DAPI'")]
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
                " example: 'z3 - sample: {i}'."
            ),
        ),
    ]


@cappa.command(name="segd")
@dataclass
class DistributedSegConf:
    img_fmt: Annotated[
        str,
        cappa.Arg(
            short="-i",
            help=(
                "pattern for input mosaic files, in python format string format."
                " following patterns assumed present: 'c' (for channel) and 'z' (for z stack level)."
                " example: 'data_dir/region_0/images/mosaic_{c}_z{z}.tif'."
            ),
        ),
    ]
    cyt_pat: Annotated[str, cappa.Arg(short="-c", help="name of cytoplasm channel. example: 'PolyT'.")]
    nuc_pat: Annotated[str, cappa.Arg(short="-n", help="name of nuclear channel. example: 'DAPI'.")]
    z_slices: Annotated[
        list[int],
        cappa.Arg(
            short="-z",
            action=cappa.ArgAction("append"),
            help="z slices to consider for segmentation. can be provided multiple times to specify multiple slices.",
        ),
    ]
    out_path: Annotated[Path, cappa.Arg(short="-o", help="path for output masks file (npy or zarr)")]
    tempdir: Annotated[Path, cappa.Arg(short="-pt", help="path to temporary directory")]
    diameter: Annotated[
        int | None,
        cappa.Arg(
            short="-d",
            help="cell diameter in pixels provided to cellpose. if left unset will be determined autonomously by cellpose.",
        ),
    ] = None
    batch_size: Annotated[int, cappa.Arg(short="-b", help="batch size specified to 'Cellpose.eval'")] = 8
    model_path: Annotated[
        Path | None,
        cappa.Arg(short="-w", help="optional path to cellpose model weights to use (uses cpsam if unset)"),
    ] = None
    chunk_x: Annotated[int, cappa.Arg(short="-lx", help="chunk block x-axis length for distributed processing")] = 256
    chunk_y: Annotated[int, cappa.Arg(short="-ly", help="chunk block y-axis length for distributed processing")] = 256
    chunk_z: Annotated[int, cappa.Arg(short="-lz", help="chunk block z-axis length for distributed processing")] = 256
    cellprob_threshold: Annotated[float, cappa.Arg(short="-tc", help="cellprob_threshold specified to 'Cellpose.eval'")] = 0.0
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


@cappa.command(name="spot")
@dataclass
class SpotConf:
    dt_path: Annotated[Path, cappa.Arg(short="-i", help="input detected transcripts CSV file path")]
    ad_path: Annotated[Path, cappa.Arg(short="-o", help="output anndata h5ad file path")]
    spot_side: Annotated[float, cappa.Arg(short="-s", help="length of square spot side, in microns")]
    z_micron_distance: Annotated[float, cappa.Arg(short="-z", help="distance between z-stacks in image, in microns")]
    flatten: Annotated[
        bool,
        cappa.Arg(
            short="-f",
            action=cappa.ArgAction("store_true"),
            help="pass to ignore z-axis when generating spots (i.e. flattening the data)",
        ),
    ] = False
