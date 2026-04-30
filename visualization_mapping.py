import pandas as pd

import numpy as np

import re

food_insecurity = pd.read_csv(
    "https://raw.githubusercontent.com/RobertEBarron/Practical-Data-Science-Final-Project/main/food_insecurity_cluster_merged_df.csv"
)


food_insecurity["year"] = pd.to_datetime(food_insecurity["survey_date"]).dt.year


def normalize_region(name: str) -> str:
    """
    Lowercase, strip accents/diacritics, collapse whitespace,
    and apply manual aliases so both datasets share the same key.
    """
    import unicodedata

    # 1. Strip leading/trailing whitespace
    name = name.strip()

    # 2. Remove accents (NFD decompose → drop combining marks → recompose)
    name = unicodedata.normalize("NFD", name)
    name = "".join(ch for ch in name if unicodedata.category(ch) != "Mn")
    name = unicodedata.normalize("NFC", name)

    # 3. Lowercase
    name = name.lower()

    # 4. Replace hyphens and apostrophes with spaces for uniform tokenisation
    name = re.sub(r"[-'']", " ", name)

    # 5. Collapse multiple spaces
    name = re.sub(r"\s+", " ", name).strip()

    # 6. Manual aliases  (apply AFTER the steps above so inputs are already clean)
    aliases = {
        # Dataset 1 variants → canonical key
        "barh el ghazel": "barh el ghazel",
        # Dataset 2 variants → canonical key
        "barh el gazel": "barh el ghazel",  # Barh-El-Gazel  → Barh el Ghazel
        "ville de n djamena": "n djamena",  # Ville de N'Djamena → N'Djamena
        "n djamena": "n djamena",
    }
    return aliases.get(name, name)


import pygadm

# Chad
chad = pygadm.Items(admin="TCD", content_level=2)

# Nigeria
nigeria = pygadm.Items(admin="NGA", content_level=2)

# Niger
niger = pygadm.Items(admin="NER", content_level=2)

# Mali
mali = pygadm.Items(admin="MLI", content_level=2)


pre_fabricated_counties = [chad, nigeria, niger, mali]

pre_fabricated_counties = pd.concat(pre_fabricated_counties)


pre_fabricated_counties["NAME_2"] = pre_fabricated_counties["NAME_2"].apply(
    normalize_region
)

food_insecurity["adm2_name"] = food_insecurity["adm2_name"].apply(normalize_region)


# cleaning the different countries: Chad

mask = food_insecurity["adm0_name"] == "Chad"

food_insecurity.loc[mask, "adm2_name"] = food_insecurity.loc[
    mask, "adm2_name"
].str.replace("north kanem", "kanem", case=False)

food_insecurity.loc[mask, "adm2_name"] = food_insecurity.loc[
    mask, "adm2_name"
].str.replace("nord kanem", "kanem", case=False)

food_insecurity.loc[mask, "adm2_name"] = food_insecurity.loc[
    mask, "adm2_name"
].str.replace("mayo lemie", "kabbia", case=False)

food_insecurity.loc[mask, "adm2_name"] = food_insecurity.loc[
    mask, "adm2_name"
].str.replace("barh el gazel ouest", "barh el ghazel", case=False)
food_insecurity.loc[mask, "adm2_name"] = food_insecurity.loc[
    mask, "adm2_name"
].str.replace("barh el gazel nord", "barh el ghazel", case=False)
food_insecurity.loc[mask, "adm2_name"] = food_insecurity.loc[
    mask, "adm2_name"
].str.replace("barh el gazel sud", "barh el ghazel", case=False)

food_insecurity.loc[mask, "adm2_name"] = food_insecurity.loc[
    mask, "adm2_name"
].str.replace("gueni", "dodje", case=False)

food_insecurity.loc[mask, "adm2_name"] = food_insecurity.loc[
    mask, "adm2_name"
].str.replace("kimiti", "djourf al ahmar", case=False)

food_insecurity.loc[mask, "adm2_name"] = food_insecurity.loc[
    mask, "adm2_name"
].str.replace("abdi", "aboudeia", case=False)


# cleaning the different countries: Niger

mask = food_insecurity["adm0_name"] == "Niger"

food_insecurity.loc[mask, "adm2_name"] = food_insecurity.loc[
    mask, "adm2_name"
].str.replace("abalak", "keita", case=False)

food_insecurity.loc[mask, "adm2_name"] = food_insecurity.loc[
    mask, "adm2_name"
].str.replace("tillaberi", "tillabery", case=False)

food_insecurity.loc[mask, "adm2_name"] = food_insecurity.loc[
    mask, "adm2_name"
].str.replace("falmey", "boboye", case=False)

food_insecurity.loc[mask, "adm2_name"] = food_insecurity.loc[
    mask, "adm2_name"
].str.replace("gotheye", "kollo", case=False)


# cleaning the different countries: Nigeria


mask = food_insecurity["adm0_name"] == "Nigeria"


food_insecurity.loc[mask, "adm2_name"] = food_insecurity.loc[
    mask, "adm2_name"
].str.replace("birnin magaji", "birnin magaji/kiyaw", case=False)

food_insecurity.loc[mask, "adm2_name"] = food_insecurity.loc[
    mask, "adm2_name"
].str.replace("northern katsina 1", "katsina", case=False)

food_insecurity.loc[mask, "adm2_name"] = food_insecurity.loc[
    mask, "adm2_name"
].str.replace("central katsina 2", "katsina", case=False)

food_insecurity.loc[mask, "adm2_name"] = food_insecurity.loc[
    mask, "adm2_name"
].str.replace("central sokoto 1", "sokoto north", case=False)

food_insecurity.loc[mask, "adm2_name"] = food_insecurity.loc[
    mask, "adm2_name"
].str.replace("central sokoto 2", "sokoto south", case=False)


merged_geospatial_df = pd.merge(
    food_insecurity,
    pre_fabricated_counties,
    how="left",
    left_on="adm2_name",
    right_on="NAME_2",
    indicator=True,
)


merged_geospatial_df["counter"] = 1

collapsed_df = (
    merged_geospatial_df.groupby(["adm0_name", "adm2_name", "cluster", "year"])
    .agg(counter=("counter", "sum"), geometry=("geometry", "first"))
    .reset_index()
)


wide_df = collapsed_df.pivot(
    index=["adm0_name", "adm2_name", "year", "geometry"],
    columns="cluster",
    values="counter",
)

wide_df = wide_df.reset_index()

wide_df.columns.name = None


wide_df = wide_df.rename(
    columns={
        0: "cluster_1",
        1: "cluster_2",
        2: "cluster_3",
        3: "cluster_4",
        4: "cluster_5",
        5: "cluster_6",
        6: "cluster_7",
    }
)

wide_df = wide_df.fillna(0)


import numpy.random as npr


def get_num_points(population, people_per_point):

    integer_component = (population / people_per_point).astype("int")
    just_decimal_component = (population / people_per_point) % 1

    # random draw from 0/1 with prob from decimal
    extra = npr.binomial(1, just_decimal_component)

    num_points = integer_component + extra
    return num_points

    import geopandas as gpd


wide_df = gpd.GeoDataFrame(wide_df, geometry="geometry")


##Proportion Table

cluster_cols = [
    "cluster_1",
    "cluster_2",
    "cluster_3",
    "cluster_4",
    "cluster_5",
    "cluster_6",
    "cluster_7",
]

summary_table = wide_df.groupby(["adm0_name", "year"])[cluster_cols].sum().reset_index()

summary_table["total_hh"] = summary_table[cluster_cols].sum(axis=1)

summary_table[cluster_cols] = summary_table[cluster_cols].div(
    summary_table["total_hh"], axis=0
)


import matplotlib.pyplot as plt

colors = {
    "cluster_1": "#E41A1C",  # red
    "cluster_2": "#377EB8",  # blue
    "cluster_3": "#7B3F9E",  # purple
    "cluster_4": "#FF7F00",  # orange
    "cluster_5": "#8B4513",  # brown
    "cluster_6": "#F781BF",  # pink
    "cluster_7": "#999999",  # grey
}

countries = summary_table["adm0_name"].unique()

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for i, country in enumerate(countries):
    ax = axes[i]
    country_df = summary_table[summary_table["adm0_name"] == country]

    for col, color in colors.items():
        ax.plot(country_df["year"], country_df[col], marker="o", label=col, color=color)

    ax.set_title(country)
    ax.set_xlabel("Year")
    ax.set_ylabel("Proportion of Households")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_xticks(country_df["year"])

plt.suptitle("Cluster Proportions by Country and Year", fontsize=14, y=1.02)
plt.tight_layout()
plt.show()


# collapsing to municipality level

static_county_demographics = wide_df.copy()

agg_dict = {col: "sum" for col in cluster_cols}
agg_dict["geometry"] = "first"

static_county_demographics = (
    static_county_demographics.groupby(["adm0_name", "adm2_name"])
    .agg(agg_dict)
    .reset_index()
)

static_county_demographics["most_common_cluster"] = static_county_demographics[
    cluster_cols
].idxmax(axis=1)


static_county_demographics = gpd.GeoDataFrame(
    static_county_demographics, geometry="geometry"
)


import contextily as cx

import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8, 5))


ax = static_county_demographics.plot(
    ax=ax, column="most_common_cluster", cmap="Set1", alpha=0.4, legend=True
)

cx.add_basemap(ax, crs="EPSG:4326", alpha=0.4)
