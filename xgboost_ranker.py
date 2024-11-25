import numpy as np
import pandas as pd
from glob import glob
from xgboost import XGBRanker
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

from utils import _FAMILY_MAPPING

def rank_families(dataset_name, seed=0):
    scores = pd.read_csv("family_scores_ranked.csv")
    scores = scores.melt(id_vars=["name"], var_name="family_name", value_name="score")
    all_files = glob("embedding/*.csv")

    feautres = []
    names = []
    for file in all_files:
        meta = np.loadtxt(file, delimiter=",")
        feautres.append(meta.flatten())
        names.append(file.split("/")[-1].replace(".csv", "").replace("_R", ""))

    feautres = pd.DataFrame(feautres)
    feautres["name"] = names

    combined = pd.merge(scores, feautres, on="name")
    combined.sort_values("name", inplace=True)
    combined.columns = combined.columns.astype(str)

    combined = combined[combined["name"] != dataset_name]

    group_sizes = combined.groupby("name").size().to_numpy()
    X = combined.drop(columns=["score", "name"])
    y = combined["score"]

    cat_attribs = ["family_name"]
    full_pipeline = ColumnTransformer([('cat', OneHotEncoder(handle_unknown='ignore'), cat_attribs)], remainder='passthrough')
    X = full_pipeline.fit_transform(X)

    model = XGBRanker(random_state=seed, objective="rank:pairwise")
    model.fit(X, y, group=group_sizes)

    files = glob(f"embedding/*{dataset_name}*.csv")
    file = min(files, key=len)
    meta = np.loadtxt(file, delimiter=",")
    meta = meta.flatten().reshape(1, -1)
    meta_df = pd.DataFrame(meta)
    meta_df.columns = meta_df.columns.astype(str)
    family_names = pd.read_csv("family_scores_ranked.csv").columns[:-1]
    meta_df = meta_df.assign(key=1).merge(pd.DataFrame(family_names, columns=["family_name"]).assign(key=1), on="key").drop("key", axis=1)
    meta_df = full_pipeline.transform(meta_df)
    best = model.predict(meta_df)
    best_families = list(zip(best, family_names))
    best_families.sort()
    best_families = [_FAMILY_MAPPING[family] for _, family in best_families]
    return best_families