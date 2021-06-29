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
#     display_name: Python 3.8.5 64-bit ('usr')
#     name: python385jvsc74a57bd031f2aee4e71d21fbe5cf8b01ff0e069b9275f58929596ceb00d14d90e3e16cd6
# ---

# %% [markdown]
"""
# Diplomatura en Ciencias de Datos, Aprendizaje Automático y sus Aplicaciones

Autores: Matías Oria, Antonela Sambuceti, Pamela Pairo, Benjamín Ocampo
"""
# %% [markdown]
"""
## Definición de funciones *helper*
Inicialmente se definen funciones que se utilizaron durante la selección e
imputación de columnas del conjunto de datos obtenido en
`melbourne_exploration.py` y almacenados en un servidor para su acceso remóto.
"""
# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn
from sklearn.experimental import enable_iterative_imputer
from sklearn import (base, decomposition, feature_extraction, impute,
                     neighbors, preprocessing)
from typing import List, Tuple, Union


def plot_imputation_graph(imputations: List[Tuple[str, pd.DataFrame]],
                          missing_cols: List[str]) -> None:
    """
    Makes a group of density plots according to the number of columns on the
    dataframes inside @imputations. @imputations must be a list of pairs
    (@method_name, @value_df) where each @value_df has the same @missing_cols
    obtained by its corresponding imputer @method_name.
    """
    _, axs = plt.subplots(len(missing_cols), figsize=(10, 10))
    for ax, col_name in zip(axs, missing_cols):
        data = pd.concat([
            imputation_df[[col_name]].assign(method=method)
            for method, imputation_df in imputations
        ])
        seaborn.kdeplot(data=data, x=col_name, hue="method", ax=ax)


def impute_by(values: Union[np.array, pd.DataFrame],
              missing_col_names: List[str],
              estimator: base.BaseEstimator) -> pd.DataFrame:
    """
    Returns a dataframe that fills null entries of @values according to
    @estimator. @missing_col_names are labels that will be assigned when
    created. @values might have columns that doesn't have null values, such that
    the IterativeImputer class takes advantage of it in order to estimate
    missing values.
    """
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
Con el fin de poder entrenar un modelo bajo las variables en `melb_combined_df`,
se deben codificar aquellas que sean categóricas. Para ello, se utiliza *One-Hot
Encoding* donde se muestran dos posibles métodos distintos.
"""
# %% [markdown]
"""
### Dict Vectorizer
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
Se obtiene una matriz de $13206 \times 20$ cuyas columnas son las que se
muestran por `get_feature_names()`.

Del total de variables de `melb_combined_df` se excluyen `suburb_name` y
`suburb_council_area` con el fin de obtener una matriz cuyo espacio en memoria
no imposibilite la imputación de las variables numéricas `housing_year_built` y
`housing_building_area` en una sección posterior. A su vez, se presentó otra
forma de realizar una codificación *one-hot* con fin complementario. No
obstante, se trabajó sobre la matriz dada por el método `Dict Vectorizer`.
"""
# %% [markdown]
"""
### One-Hot Encoding
Otro forma posible de obtener la matriz de *features* es por medio de la clase
`OneHotEncoder` realizando la codificación sobre `categorical_cols`.
"""
# %%
ohe = preprocessing.OneHotEncoder(sparse=False)
feature_matrix_ohe = np.hstack([
    ohe.fit_transform(melb_combined_df[categorical_cols]),
    melb_combined_df[numerical_cols]
])
feature_matrix_ohe.shape
# %% [markdown]
"""
De manera similar, se obtiene la matriz codificada donde las primeras 17
columnas corresponden a la codificación de las variables categóricas, y las 3
restantes a las numéricas que no requerían este paso.
"""
# %% [markdown]
"""
## Imputación por KNN
En esta sección se procedió a imputar las variables `housin_year_built` y
`housing_building_area` con el fin de incluirlas en la matriz de *features*.
Para ello, se realizaron imputaciones univariadas y multivariadas, es decir,
utilizando solamente valores no nulos de dichas variables para realizar la
imputación, en comparación a, aprovechar todos los *features* disponibles en la
matriz obtenida en la sección anterior. A su vez, esta comparación también se
realizó estandarizando las variables con el fin de verificar si influía en la
imputación resultante.
"""
# %% [markdown]
"""
### Sin estandarizado
"""
# %%
missing_cols = ["housing_year_built", "housing_building_area"]
estimator = neighbors.KNeighborsRegressor(n_neighbors=2)
# %%
missing_df = melb_combined_df[missing_cols]
original_df = missing_df.dropna()
all_df = np.hstack([missing_df, feature_matrix.todense()])

knn_missing_cols = impute_by(missing_df, missing_cols, estimator)
knn_all_cols = impute_by(all_df, missing_cols, estimator)
# %% [markdown]
"""
Para la comparación se crean 3 *dataframes*:
  - `original_df`: Contiene las columnas de `housing_year_built` y
    `housing_building_area` eliminando aquellos valores faltantes.
  - `missing_df`: Similar a `original_df` pero sin eliminar las entradas nulas.
  - `all_df`: Contiene todas las *features* obtenidas en la sección de
    codificación junto a las de `missing_df`.

Posteriormente, se procedió a imputar los valores faltantes que ocurren en las
entradas de `missing_df` y `all_df` por medio de `impute_by`, una de las
funciones *helper* definidas en la primera sección, generando un nuevo
*dataframe* con aquellos datos completados con el estimador `KNeighbors`.

Por último ,las distribuciones de las observaciones de los *dataframes*
resultantes se comparan por medio de un gráfico de densidad.
"""
# %%
imputations = [
    ("original", original_df),
    ("knn - missing cols", knn_missing_cols),
    ("knn - all cols", knn_all_cols)
]
plot_imputation_graph(imputations, missing_cols)
# %% [markdown]
"""
Se puede observar que la destribución de `housing_year_built` luego de imputar
por medio de todas las columnas captura en mejor medida la tendencia
correspondiente a su original. Especialmente para aquellas viviendas con poca
antiguedad al momento de la venta. Similarmente, puede darse la misma
aceberación para la variable `housing_bulding_area`.
"""
# %% [markdown]
"""
### Con estandarizado
"""
# %%
scaler = preprocessing.StandardScaler()
original_scaled_df = pd.DataFrame(scaler.fit_transform(original_df),
                                  columns=missing_cols)
knn_scaled_missing_cols = impute_by(scaler.fit_transform(missing_df),
                                    missing_cols, estimator)
knn_scaled_all_cols = impute_by(scaler.fit_transform(all_df), missing_cols,
                                estimator)
# %%
imputations = [
    ("scaled original", original_scaled_df),
    ("knn - scaled missing cols", knn_scaled_missing_cols),
    ("knn - scaled all cols", knn_scaled_all_cols)
]
plot_imputation_graph(imputations, missing_cols)
# %% [markdown]
"""
Escalando los datos de manera previa a la imputación no pareció afectar la
elección de un método por sobre otro. Se puede observar que la manera en que se
distribuyen ambas variables es similar al caso anterior salvo por el cambio de
escala en los ejes x e y. Por lo tanto, se optó por incluir la imputación con
todas las *features* sin estandarizado previo a la matriz resultante de la
sección anterior.
"""
# %%
feature_matrix = np.hstack([feature_matrix.todense(), knn_all_cols])
feature_matrix.shape
# %% [markdown]
"""
## Reducción de dimensionalidad
"""
# %% [markdown]
"""
### Análisis de Componentes Principales (PCA)
Antes de realizar el PCA se realiza la estandarización de los datos, es decir a
cada dato se le resta su media y se lo divide por el desvío estandar. La
estandarización permite trabajar con variables medidas en distintas unidades y
así dar el mismo peso a todas las variables.
"""
# %%
feature_matrix_standarized = preprocessing.StandardScaler().fit_transform(
    feature_matrix)
# %% [markdown]
"""
A continuación se muestra a modo de ejemplo el cambio de los valores antes y
después de la estandarización para la fila 6.
"""
# %%
print('\nAntes de estandarizar \n%s' %feature_matrix[5])
print('\nDespués de estandarizar \n%s' %feature_matrix_standarized[5])
# %%
pca = decomposition.PCA(n_components=22)
principalComponents = pca.fit_transform(feature_matrix_standarized)
# %% [markdown]
"""
Se muestra la varianza explicada por cada componente
"""
# %%
exp_var_pca = pca.explained_variance_ratio_
exp_var_pca
# %% [markdown]
"""
Para una mejor comprensión se calcula el porcentaje acumulado explicado por cada
componente. El primer componente explica el 17,09% de la variación. Luego, el
primer y segundo componente explican el 27.32% de la variación y asi
sucesivamente.
"""
# %%
var=np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)
var
# %% [markdown]
"""
Se eligen las primeras 18 componentes que explican el 98,71% de la variación y
también a partir de ese valor se llega al plateau.
"""
# %%
plt.figure(figsize=(10, 5))
plt.ylabel('% de varianza explicada')
plt.xlabel('Componentes Principales')
plt.title('PCA')
plt.ylim(20, 110)
plt.xticks(range(0, 23))
plt.style.context('seaborn-whitegrid')
plt.plot(var)
# %% [markdown]
"""
El siguiente gráfico muestra en conjunto la varianza explicada por cada
componente (barra) y la varianza acumulada (linea escalonada).
"""
# %%
cum_sum_eigenvalues = np.cumsum(exp_var_pca)
# %%
plt.figure(figsize=(10, 5))
plt.bar(range(0, len(exp_var_pca)),
        exp_var_pca,
        alpha=0.5,
        align='center',
        label='Varianza explicada individual')
plt.step(range(0, len(cum_sum_eigenvalues)),
         cum_sum_eigenvalues,
         where='mid',
         label='Varianza explicada acumulada')
plt.ylabel('Explained variance ratio')
plt.xlabel('Componentes principales')
plt.legend(loc='best')
plt.xticks(range(0, 23))
plt.tight_layout()
plt.show()
# %% [markdown]
"""
## Composición del resultado
"""
