import geopandas as gpd
import numpy as np
import pandas as pd
from anndata import AnnData
from scipy.sparse import csr_matrix
from shapely import box

from ..conf import SpotConf


def sjts(tdf: gpd.GeoDataFrame, sdf: gpd.GeoDataFrame) -> pd.DataFrame:
    return (
        tdf.sjoin(
            sdf,
            predicate="within",
        )[["index_right", "coords", "index_transcript", "gene"]]
        .merge(
            sdf.rename_geometry("spot"),
            left_on="index_right",
            right_index=True,
        )
        .assign(
            distance=lambda x: x["spot"].centroid.distance(x["coords"]),
            label=lambda x: x["index_right"].astype("Int64"),
        )
        .sort_values("distance")
        .drop_duplicates("index_transcript")
    )


def run(conf: SpotConf):
    print(f"loading detected transcripts table from {conf.dt_path}", flush=True)
    tdf = pd.read_csv(
        conf.dt_path,
        usecols=["gene", "global_x", "global_y", "global_z"],  # pyright: ignore
    )
    tdf = (
        gpd.GeoDataFrame(tdf[["gene", "global_z"]], geometry=gpd.points_from_xy(tdf["global_x"], tdf["global_y"]))
        .rename_geometry("coords")
        .reset_index(names="index_transcript")
    )

    bbox_minx, bbox_miny, bbox_maxx, bbox_maxy = tdf.total_bounds

    scale_factor = 0.02
    sdf = gpd.GeoDataFrame(
        gpd.GeoSeries(
            [
                box(
                    xmin - (conf.spot_side * (scale_factor / 2)),
                    ymin - (conf.spot_side * (scale_factor / 2)),
                    xmin + (conf.spot_side * (1 + scale_factor)),
                    ymin + (conf.spot_side * (1 + scale_factor)),
                )
                for ymin in np.arange(bbox_miny - conf.spot_side / 2, bbox_maxy + conf.spot_side / 2, conf.spot_side)
                for xmin in np.arange(bbox_minx - conf.spot_side / 2, bbox_maxx + conf.spot_side / 2, conf.spot_side)
            ]
        ).rename("coords"),
        geometry="coords",
    )
    if conf.flatten:
        print("running spatial join between transcripts and spots", flush=True)
        jdf = sjts(
            tdf,  # pyright: ignore
            sdf,
        )
    else:
        jdf = pd.DataFrame()
        for z, tdf_slice in tdf.groupby("global_z"):
            print(f"z={z}: running spatial join between transcripts and spots", flush=True)
            jdf = pd.concat(
                [
                    jdf,
                    sjts(
                        tdf_slice,  # pyright: ignore
                        sdf,
                    ).assign(z=z),
                ]
            )

    # sanity check assignments
    assert jdf.index.size == tdf.index.size, "bug: not all transcripts were assigned a spot"

    if not conf.flatten:
        jdf["label"] = jdf["z"].astype(str) + "_" + jdf["label"].astype(str)

    print("constructing count matrix", flush=True)
    mtx = (
        jdf.groupby(["label", "gene"])["gene"]
        .count()
        .to_frame(name="n")
        .reset_index()
        .pivot(index="label", columns="gene", values="n")
        .fillna(0)
        .astype(int)
    )

    print("constructing anndata object", flush=True)

    ad = AnnData(mtx)

    # remove blanks from gene matrix, keep as obsm slot
    blank_filter = ad.var_names.str.startswith("Blank-")
    ad.obsm["blanks"] = pd.DataFrame(
        ad[:, blank_filter].X, index=ad.obs_names, columns=ad.var_names[blank_filter]  # pyright: ignore
    )
    ad = ad[:, ~blank_filter].copy()

    # switch to CSR matrix data storage
    ad.X = csr_matrix(ad.X)

    # add spatial information
    if not conf.flatten:
        spatial = sdf.merge(
            ad.obs_names.to_series()
            .str.split("_", n=1, expand=True)
            .assign(z=lambda x: x[0].astype(float) * conf.z_micron_distance, sdf_label=lambda x: x[1].astype(int))[
                ["z", "sdf_label"]
            ],
            left_index=True,
            right_on="sdf_label",
        ).loc[ad.obs_names]
        ad.obs["spot_z"] = spatial["z"]
    else:
        spatial = sdf.loc[ad.obs_names]
    ad.obs["spot_x"] = spatial.geometry.centroid.x
    ad.obs["spot_y"] = spatial.geometry.centroid.y

    print(f"saving anndata to {conf.ad_path}", flush=True)
    ad.write_h5ad(conf.ad_path)
    ad.write_h5ad(conf.ad_path)
