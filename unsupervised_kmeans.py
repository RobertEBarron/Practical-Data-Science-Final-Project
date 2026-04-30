import numpy as np

import pandas as pd

import patsy

food_insecurity = pd.read_csv(
    "https://raw.githubusercontent.com/RobertEBarron/Practical-Data-Science-Final-Project/main/diem_features.csv"
)


import re

food_insecurity.columns = [re.sub(" ", "_", c) for c in food_insecurity.columns]
food_insecurity.columns = [re.sub("[%/()-]", "", c) for c in food_insecurity.columns]


X = patsy.dmatrix(
    "fies_rawscore + lcsi + shock_noshock + shock_sicknessordeathofhh + shock_lostemplorwork + shock_otherintrahhshock + shock_higherfoodprices + shock_higherfuelprices + shock_mvtrestrict + shock_othereconomicshock + shock_pestoutbreak + shock_plantdisease + shock_animaldisease + shock_napasture + shock_othercropandlivests + shock_coldtemporhail + shock_flood + shock_drought + shock_othernathazard + shock_violenceinsecconf + shock_theftofprodassets + shock_othermanmadehazard + female_headed + hh_size_ordinal + edu_none + edu_primary + edu_secondary + edu_higher + edu_religious_informal + agric_crop_only + agric_livestock_only + agric_both + agric_none + toilet_flush + toilet_pit_latrine + toilet_open_pit + toilet_communal + toilet_other + toilet_none_bush",
    data=food_insecurity,
    return_type="dataframe",
)


import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import silhouette_score

from sklearn.cluster import KMeans

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


inertias = []
silhouette_scores = []
k_range = range(2, 11)

for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    inertias.append(km.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, labels))


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(k_range, inertias, "bo-")
ax1.set_xlabel("Number of clusters (k)")
ax1.set_ylabel("Inertia")
ax1.set_title("Elbow Method")


ax2.plot(k_range, silhouette_scores, "ro-")
ax2.set_xlabel("Number of clusters (k)")
ax2.set_ylabel("Silhouette Score")
ax2.set_title("Silhouette Score")


plt.tight_layout()
plt.savefig(
    r"C:\Users\mikeb\OneDrive - Duke University\Desktop\Duke\Year 2\Semester 2\Classes\Practical Data Science\Final Project\cleaned_data\elbow_silhouette.png",
    dpi=300,
    bbox_inches="tight",
)


# fit final model
km_final = KMeans(n_clusters=7, random_state=42, n_init=10)
food_insecurity["cluster"] = km_final.fit_predict(X_scaled)


identifiers = pd.read_csv(
    "https://raw.githubusercontent.com/RobertEBarron/Practical-Data-Science-Final-Project/main/diem_identifiers.csv"
)


merged_df = pd.concat([food_insecurity, identifiers], axis=1)


# merged_df.to_csv()
