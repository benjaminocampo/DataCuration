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
# %% [markdown]
"""
## Enconding 
"""
# %% [markdown]
"""
### OneHotEncodeding
"""
# %%
import numpy as np
import pandas as pd
from sklearn import preprocessing, impute, neighbors
from sklearn.experimental import enable_iterative_imputer

URL_MELB_HOUSING_FILTERED = "https://www.famaf.unc.edu.ar/~nocampo043/melb_housing_filtered_df.csv"
URL_MELB_SUBURB_FILTERED = "https://www.famaf.unc.edu.ar/~nocampo043/melb_suburb_filtered_df.csv"

melb_housing_df = pd.read_csv(URL_MELB_HOUSING_FILTERED)
melb_suburb_df = pd.read_csv(URL_MELB_SUBURB_FILTERED)

# %%
melb_combim = melb_housing_df.join(melb_suburb_df, on="suburb_id")
melb_combim[:5]

# %%
categorical_cols = ["housing_room_segment", "housing_bathroom_segment", "housing_type", "suburb_name","suburb_region_segment", "suburb_council_area"]
numerical_cols = ["housing_price","housing_land_size","suburb_rental_dailyprice"]

# %%
print("housing_room_segment", melb_combim.housing_room_segment.nunique())
print("housing_bathroom_segment", melb_combim.housing_bathroom_segment.nunique())
print("housing_type", melb_combim.housing_type.nunique())
print("suburb_region_segment", melb_combim.suburb_region_segment.nunique())
print("suburb_council_area", melb_combim.suburb_council_area.nunique())
print("suburb_name", melb_combim.suburb_name.nunique())

# %%
encoder = OneHotEncoder(sparse=False)
encoder.fit(melb_combim[categorical_cols])
# We can inspect the categories found by the encoder
encoder.categories_

# %%
encoded_types = encoder.transform(melb_combim[categorical_cols])
encoded_types[:10]

# %%
melb_numeric = melb_combim[numerical_cols].values
melb_numeric[:10]

# %%
matriz = np.hstack((encoded_types, melb_numeric))
matriz [:5]

# %%
matriz.shape

# %% [markdown]
"""
### DictVectorizer
"""
# %%
feature_cols = ["housing_room_segment", "housing_bathroom_segment", "housing_type", "suburb_name","suburb_region_segment", "suburb_council_area",
"housing_price","housing_land_size","suburb_rental_dailyprice"]
feature_dict = list(melb_combim[feature_cols].T.to_dict().values())
feature_dict[:2]
# %%
vec = DictVectorizer()
feature_matrix = vec.fit_transform(feature_dict)
# %%
feature_matrix
# %%
vec.get_feature_names()
# %% [markdown]
"""
## Imputación por KNN
"""
# %%
at_least_onena= melb_combim['housing_year_built'].isna() | melb_combim['housing_building_area'].isna()
missing_values_indices= melb_combim[at_least_onena].index
melb_combim[at_least_onena]
# %%
mice_imputer = IterativeImputer(random_state=0, estimator=KNeighborsRegressor(n_neighbors=2, weights="uniform"))

melb_combim.loc[:,['housing_year_built_imputer', 'housing_building_area_imputer']] = mice_imputer.fit_transform(
    melb_combim[['housing_year_built', 'housing_building_area']])
# %%
mice_year_built_knn = melb_combim["housing_year_built_imputer"].to_frame().rename(columns={"housing_year_built_imputer":"housing_year_built"})
mice_year_built_knn['Imputation'] = 'KNN over YearBuilt'
melb_year_built_orig = melb_combim["housing_year_built"].dropna().to_frame()
melb_year_built_orig['Imputation'] = 'Original'
data = pd.concat([mice_year_built_knn, melb_year_built_orig])
fig = plt.figure(figsize=(8, 5))
g = seaborn.kdeplot(data=data, x='housing_year_built', hue='Imputation')
# %%
mice_build_area_knn = melb_combim["housing_building_area_imputer"].to_frame().rename(columns={"housing_building_area_imputer":"housing_building_area"})
mice_build_area_knn['Imputation'] = 'KNN over Building Area'
melb_build_area_orig = melb_combim["housing_building_area"].dropna().to_frame()
melb_build_area_orig['Imputation'] = 'Original'
data = pd.concat([mice_build_area_knn, melb_build_area_orig])
fig = plt.figure(figsize=(8, 5))
g = seaborn.kdeplot(data=data, x="housing_building_area", hue='Imputation')
# %%
print(melb_year_built_orig)
