import gzip

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely as shp
import zarr
from skimage.draw import polygon2mask

from ..conf import FromProsegConf


def r2m(cell: int | np.integer, geom: shp.MultiPolygon, shape: tuple[int, int]) -> np.ndarray:
    val_polys = []
    for pol in geom.geoms:
        pol = shp.make_valid(pol)
        assert isinstance(pol, shp.Polygon), f"expected Polygon when iterating over MultiPolygon, got {pol.wkt}"
        val_polys.append(polygon2mask(shape, np.array(pol.exterior.coords)[:, [1, 0]]))

    return (np.bitwise_or.reduce(val_polys, axis=0) * cell).astype(int)


def process_zslice(df: pd.DataFrame, shape: tuple[int, int]) -> np.ndarray:
    row_iter = df.itertuples()
    row = next(row_iter)
    # for type cheker
    assert isinstance(row.cell, int | np.integer)
    assert isinstance(row.geometry, shp.MultiPolygon)
    masks = r2m(row.cell, row.geometry, shape)

    for row in row_iter:
        # for type checker
        assert isinstance(row.cell, int | np.integer)
        assert isinstance(row.geometry, shp.MultiPolygon)

        geom = row.geometry
        x_min, y_min, x_max, y_max = map(round, geom.bounds)

        # create mask over bbox-bounded view of full z-slice
        curr_mask = r2m(row.cell, shp.affinity.translate(geom, -x_min, -y_min), (y_max - y_min + 1, x_max - x_min + 1))

        # add mask (accounting for overlapping masks) to full view
        masks[y_min : (y_max + 1), x_min : (x_max + 1)] = (
            masks[y_min : (y_max + 1), x_min : (x_max + 1)] * (curr_mask == 0)
        ) + curr_mask

    return masks


def run(conf: FromProsegConf):
    print(f"loading micron to pixel transform from {conf.mp_path}", flush=True)
    tfm = np.genfromtxt(conf.mp_path)[[0, 0, 1, 1, 0, 1], [0, 1, 0, 1, 2, 2]].tolist()

    print(f"loading proseg-generated cell polygons from {conf.geojson_path}", flush=True)
    gdf = gpd.read_file(
        gzip.open(conf.geojson_path) if conf.geojson_path.suffix == ".gz" else conf.geojson_path,
        use_arrow=True,
    )

    if conf.z_slice is not None:
        gdf = gdf.loc[gdf.layer == conf.z_slice]

    gdf.geometry = gdf.geometry.affine_transform(tfm)

    # crop to size of image
    gdf = gdf[gdf.within(shp.box(0, 0, conf.x_shape, conf.y_shape))]

    if conf.z_slice is None:
        stacks = []
        for i, gdf_slice in gdf.groupby("layer", sort=True):
            print(f"z={i}: computing masks", flush=True)
            stacks.append(process_zslice(gdf_slice, (conf.y_shape, conf.x_shape)))

        print(f"generating 3D stack from {len(stacks)} detected z-slices", flush=True)
        masks = np.stack(stacks, axis=0)
    else:
        print(f"z={conf.z_slice}: computing masks", flush=True)
        masks = process_zslice(gdf, (conf.y_shape, conf.x_shape))

    print(f"saving {'3' if conf.z_slice is None else '2'}D masks file to {conf.out_path}", flush=True)
    if conf.out_path.suffix == ".zarr":
        zarr.save_array(str(conf.out_path), masks)
    else:
        with open(conf.out_path, "wb") as nf:
            np.save(nf, masks)
