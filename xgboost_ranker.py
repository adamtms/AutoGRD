import numpy as np
import pandas as pd
from glob import glob
from xgboost import XGBRanker
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

from utils import _FAMILY_MAPPING, YamlWriter

scores = pd.read_csv("family_scores_ranked.csv")
scores = scores.melt(id_vars=["name"], var_name="family_name", value_name="score")
embedding_folder = "/Users/adamtomys/Projects/AutoGRD/embedding"
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

group_sizes = combined.groupby("name").size().to_numpy()
X = combined.drop(columns=["score", "name"])
y = combined["score"]


cat_attribs = ["family_name"]
full_pipeline = ColumnTransformer([('cat', OneHotEncoder(handle_unknown='ignore'), cat_attribs)], remainder='passthrough')
X = full_pipeline.fit_transform(X)


with YamlWriter("autoGRD") as writer:
    for seed in range(6):
        writer.increase_indent(seed)
        writer.increase_indent(30)
        model = XGBRanker(random_state=seed, objective="rank:pairwise")
        model.fit(X, y, group=group_sizes)

        for file in all_files:
            meta = np.loadtxt(file, delimiter=",")
            meta = meta.flatten().reshape(1, -1)
            name = file.split("/")[-1].replace(".csv", "").replace("_R", "")
            meta_df = pd.DataFrame(meta)
            meta_df.columns = meta_df.columns.astype(str)
            family_names = ["Bagging_(BAG)", "Bayesian_Methods_(BY)", "Boosting_(BST)", "Decision_Trees_(DT)", "Discriminant_Analysis_(DA)", "Generalized_Linear_Models_(GLM)",
                            "Logistic_and_Multinomial_Regression_(LMR)", "Multivariate_Adaptive_Regression_Splines_(MARS)", "Nearest_Neighbor_Methods_(NN)", "Neural_Networks_(NNET)",
                            "Other_Ensembles_(OEN)", "Other_Methods_(OM)", "Partial_Least_Squares_and_Principal_Component_Regression_(PLSR)", "Random_Forests_(RF)", "Rule-Based_Methods_(RL)",
                            "Stacking_(STC)", "Support_Vector_Machines_(SVM)"]
            meta_df = meta_df.assign(key=1).merge(pd.DataFrame(family_names, columns=["family_name"]).assign(key=1), on="key").drop("key", axis=1)
            print(meta_df["family_name"])
            meta_df = full_pipeline.transform(meta_df)
            best = model.predict(meta_df)
            best_families = list(zip(best, family_names))
            best_families.sort()
            best_families = [_FAMILY_MAPPING[family].name for _, family in best_families]
            writer.add_partial_result(name, best_families)
        writer.decrease_indent()
        writer.decrease_indent()