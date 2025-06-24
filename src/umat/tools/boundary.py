from functools import partial
from multiprocessing import Pool
from pathlib import Path
from typing import Callable

import geopandas as gpd
import numpy as np
import pandas as pd
import zarr
from shapely import MultiPolygon, Polygon, union_all
from shapely.affinity import affine_transform
from shapely.validation import explain_validity
from skimage.measure import find_contours, regionprops_table

from ..conf import BoundaryConf


def process_cell(
    z_slice: np.ndarray,
    tfm: list[float],
    props: tuple[np.int64, np.int64, np.int64, np.int64, np.int64],
) -> tuple[np.int64, MultiPolygon | None]:
    label, min_r, min_c, max_r, max_c = props

    # adding padding for find_countours
    lab_arr = np.pad(
        z_slice[min_r:max_r, min_c:max_c] == label,
        1,
        "constant",
        constant_values=False,
    )

    # generate list of polygons by:
    # - finding contour in bbox sub-region
    # - returning to global array coordinates,
    # - transforming into a valid shapely polygon (buffer to remove invalidating self-intersections)
    # early return None instead of geometry if:
    # - find_contours fails
    # - any of the Polygon constructions fails
    # - resulting geometry is empty
    try:
        polys = [
            Polygon(
                (arr + [min_r, min_c])[
                    :,  #    inversion of second axis necessary since skimage.measure.find_contourds returns array of (row,column) points,
                    ::-1,  # which corresponds to (y,x) points, but shapely.Polygon constructor expects an array of (x,y) points
                ],
            ).buffer(0)
            for arr in find_contours(lab_arr, 0.5)
        ]
    except ValueError:
        return (label, None)

    # generate union of all contour polygons
    mp = union_all(polys)

    # early return None if resulting geometry is empty
    if mp.is_empty:
        return (label, None)

    # transform from pixel coordinates to real coordinates
    # (buffer to remove invalidating self-intersections)
    mp = affine_transform(mp, tfm).buffer(0)

    # if resulting geometry is one (1) Polygon turn it into a MultiPolygon
    if isinstance(mp, Polygon):
        mp = MultiPolygon([mp])

    # sanity check: make sure we have a valid, non empty MultiPolygon, report otherwise
    assert isinstance(mp, MultiPolygon), f"expected MultiPolygon, have: {mp.wkt}"
    assert mp.is_valid, f"MultiPolygon invalid, reason: {explain_validity(mp)}"
    assert not mp.is_empty, f"got empty MultiPolygon, wkt: {mp.wkt}"

    return (label, mp)


def mk_table(z_slice: np.ndarray, z_idx: int, tfm: list[float], ncpus: int) -> pd.DataFrame:
    print(f"z={z_idx}: determining region properties", flush=True)
    props = list(zip(*regionprops_table(z_slice, properties=["label", "bbox"]).values()))

    print(f"z={z_idx}: determining cell polygons", flush=True)

    # TODO: figure out why using multiprocessing doesn't provide any speedup
    if ncpus > 1:
        with Pool(ncpus) as p:
            o = p.map(
                partial(process_cell, z_slice, tfm),
                props,
                chunksize=round(len(props) / ncpus),
            )
    else:
        o = list(map(partial(process_cell, z_slice, tfm), props))

    print(f"z={z_idx}: saving cell polygons to table", flush=True)

    return pd.DataFrame(
        {
            k: v
            for k, v in zip(
                ("label", "coords"),
                # filter out elements where processing failed
                zip(*filter(lambda t: t[1] is not None, o)),
            )
        }
        | {"global_z": z_idx}
    )


def mk_get(inp_path: Path) -> tuple[int, Callable[[int], np.ndarray]]:
    if inp_path.suffix == ".zarr":
        arr = zarr.open(str(inp_path), mode="r")
        assert isinstance(arr, zarr.Array), f"expected input file {inp_path} to contain zarr.Array, got {type(arr)}"
        get_fn = lambda z: arr[z, :, :]  # noqa: E731
    else:
        arr = np.load(inp_path, mmap_mode="r")
        get_fn = lambda z: np.copy(arr[z, :, :])  # noqa: E731

    return arr.shape[0], get_fn


def run_boundaries(conf: BoundaryConf):
    print(f"loading micron to pixel transform from {conf.mp_path}", flush=True)
    tfm = np.linalg.inv(np.genfromtxt(conf.mp_path))[[0, 0, 1, 1, 0, 1], [0, 1, 0, 1, 2, 2]].tolist()

    max_z, get = mk_get(conf.inp_path)
    cdf = pd.DataFrame()
    for z_idx in range(max_z) if conf.z_subset is None else conf.z_subset:
        print(f"z={z_idx}: slicing 2D z slice of masks from {conf.inp_path}", flush=True)
        z_slice = get(z_idx)

        cdf = pd.concat([cdf, mk_table(z_slice, z_idx, tfm, conf.ncpus)])

    print(f"saving cell table to {conf.out_path}", flush=True)
    gpd.GeoDataFrame(cdf, geometry="coords").to_feather(conf.out_path)
    gpd.GeoDataFrame(cdf, geometry="coords").to_feather(conf.out_path)
    gpd.GeoDataFrame(cdf, geometry="coords").to_feather(conf.out_path)
