import geopandas as gpd
import pandas as pd
from anndata import AnnData
from scipy.sparse import csr_array

from ..conf import AssignConf


def run(conf: AssignConf):
    print(f"loading cell boundary tables from {conf.b_paths}", flush=True)
    cdf = gpd.GeoDataFrame(pd.concat(map(gpd.read_feather, conf.b_paths), ignore_index=True), geometry="coords")

    print(f"loading detected transcripts table from {conf.dt_path}", flush=True)
    tdf = pd.read_csv(
        conf.dt_path,
        usecols=["gene", "global_x", "global_y", "global_z"],  # pyright: ignore
    )
    tdf = (
        gpd.GeoDataFrame(tdf[["gene", "global_z"]], geometry=gpd.points_from_xy(tdf["global_x"], tdf["global_y"]))
        .rename_geometry("coords")
        .reset_index(names="index_transcript")  # pyright: ignore
    )

    print("running spatial join between cells and transcripts", flush=True)
    jdf = (
        tdf.sjoin(
            cdf,
            on_attribute="global_z",  # pyright: ignore
            predicate="within",
        )[["index_right", "coords", "index_transcript", "gene"]]
        .merge(
            cdf.rename_geometry("cell"),
            left_on="index_right",
            right_index=True,
        )
        .assign(
            distance=lambda x: x["cell"].centroid.distance(x["coords"]),
            label=lambda x: x["label"].astype("Int64"),
        )
        .sort_values("distance")
        .drop_duplicates("index_transcript")
    )

    print(f"saving assigned transcript table to {conf.ft_path}")
    tdf.merge(jdf[["index_transcript", "label"]], how="left", on="index_transcript").set_index(
        "index_transcript"
    ).sort_index().rename_axis(index=None).to_feather(conf.ft_path)

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
        ad[:, blank_filter].X,  # pyright: ignore
        index=ad.obs_names,
        columns=ad.var_names[blank_filter],
    )
    ad = ad[:, ~blank_filter].copy()

    # switch to CSR matrix for data storage
    ad.X = csr_array(ad.X)

    print(f"saving anndata to {conf.ad_path}", flush=True)
    ad.write_h5ad(conf.ad_path)
