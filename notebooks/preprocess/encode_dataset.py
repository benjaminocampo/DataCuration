# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     cell_markers: '"""'
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.3
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# %% [markdown]
"""
# Diplomatura en Ciencias de Datos, Aprendizaje Automático y sus Aplicaciones

Autores: Matías Oria, Antonela Sambuceti, Pamela Pairo, Benjamín Ocampo
"""
# %%
from typing import List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn
from sklearn import preprocessing, impute, neighbors, feature_extraction, pipeline
from sklearn.experimental import enable_iterative_imputer


def plot_imputation_graph(imputations: Tuple[str, List[pd.DataFrame]],
                          missing_cols: List[str]) -> None:
    _, axs = plt.subplots(len(missing_cols), figsize=(10, 10))
    for ax, col_name in zip(axs, missing_cols):
        data = pd.concat([
            imputation_df[[col_name]].assign(method=method)
            for method, imputation_df in imputations
        ])
        seaborn.kdeplot(data=data, x=col_name, hue="method", ax=ax)


def impute_by(values, missing_col_names, estimator):
    indicator = impute.MissingIndicator()
    indicator.fit_transform(values)

    imputer = impute.IterativeImputer(
        random_state=0, estimator=estimator)
    imputed_values = imputer.fit_transform(values)
    imputed_df = pd.DataFrame(imputed_values[:, indicator.features_],
                              columns=missing_col_names)
    return imputed_df
# %%
URL_MELB_HOUSING_FILTERED = "https://www.famaf.unc.edu.ar/~nocampo043/melb_housing_filtered_df.csv"
URL_MELB_SUBURB_FILTERED = "https://www.famaf.unc.edu.ar/~nocampo043/melb_suburb_filtered_df.csv"

melb_housing_df = pd.read_csv(URL_MELB_HOUSING_FILTERED)
melb_suburb_df = pd.read_csv(URL_MELB_SUBURB_FILTERED)
melb_combined_df = melb_housing_df.join(melb_suburb_df, on="suburb_id")
melb_combined_df
# %% [markdown]
"""
## Enconding 
"""
# %%
categorical_cols = [
    "housing_room_segment", "housing_bathroom_segment", "housing_type",
    "suburb_region_segment"
]
numerical_cols = [
    "housing_price", "housing_land_size", "suburb_rental_dailyprice"
]
feature_cols = categorical_cols + numerical_cols
features = list(melb_combined_df[feature_cols].T.to_dict().values())

vectorizer = feature_extraction.DictVectorizer()
feature_matrix = vectorizer.fit_transform(features)
feature_matrix
# %%
vectorizer.get_feature_names()
# %% [markdown]
"""
## Imputación por KNN
"""
# %%
missing_cols = ["housing_year_built", "housing_building_area"]
missing_df = melb_combined_df[missing_cols]
all_df = np.hstack([missing_df, feature_matrix.todense()])
estimator = neighbors.KNeighborsRegressor(n_neighbors=2)
# %%
knn_missing_cols = impute_by(missing_df, missing_cols, estimator)
knn_all_cols = impute_by(all_df, missing_cols, estimator)
# %%
imputations = [
    ("original", missing_df.dropna()),
    ("knn - missing cols", knn_missing_cols),
    ("knn - all cols", knn_all_cols)
]
plot_imputation_graph(imputations, missing_cols)
# %% [markdown]
"""
## Reducción de dimensionalidad
"""
# %% [markdown]
"""
## Composición del resultado
"""
