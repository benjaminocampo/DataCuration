# %% [markdown]
# # Diplomatura en Ciencas de Datos, Aprendizaje Automático y sus Aplicaciones
#
# Autores: Matias Oria, Antonela Sambuceti, Pamela Pairo, Benjamín Ocampo %%
# %% [markdown]
# ## Introducción
# Se trabajó sobre el conjunto de datos de [la compentencia
# Kaggle](https://www.kaggle.com/dansbecker/melbourne-housing-snapshot) para la
# estimación de precios de ventas de propiedades en Melbourne, Australia.
#
# Este fue producido por [DanB](https://www.kaggle.com/dansbecker) de datos
# provenientes del sitio [Domain.com.au](https://www.domain.com.au/) y se
# accedió a el a través de una versión preprocesada en
# `normalize_melbourne_dataset.ipynb` siendo alojada en un servidor de la
# Universidad Nacional de Córdoba para facilitar su acceso remoto.
#
# La exploración fue realizada principalmente por medio de
# [Pandas](https://pandas.pydata.org/) y librerias de manipulación de datos.
#
# TODO: Agregar información de resultados o cosas que se buscaron al finalizar
# el proyecto.
# %% [markdown]
# ## Definición de constantes y funciones *helper*
# %%
import pandas as pd
import numpy as np
import seaborn
import matplotlib.pyplot as plt

def clean_outliers(df, column_name):
    col = df[column_name]
    mask_outlier = np.abs(col - col.mean()) <= (2.5 * col.std())
    return df[mask_outlier], df[~mask_outlier]
# %% [markdown]
# ## Elección de variables relevantes
# Dado que el objetivo es estimar el precio de ventas de viviendas en Melbourne
# es de interés analizar cuales son las variables que realmente influyen o son
# relevantes para esta tarea. Se excluyeron únicamente la dirección de las
# viviendas y su ubicación en términos de latitud y longitud dados por las
# columnas `housing_address`, `housing_bathroom2` respectivamente, dado que no
# proveen información relevante sobre el entorno en el que se encuentran. Si
# bien estos datos permiten conocer la ubicación exacta de una propiedad, no
# brindan información sobre aspectos de calidad que motiven la subida o bajada
# del precio de una vivienda en Melbourne. Son datos únicos que primero deberían
# ser agrupados en zonas para argumentar algún análisis. Agrupamientos que ya
# son brindados por el conjunto de datos por medio de otras variables que si se
# están incluyendo como el nombre del suburbio, departamento, o el código
# postal.
#
# El resto de caracteristicas de las propiedades se considera que podrían llegar
# a influenciar en el precio ya que son datos comunes que mayormente se
# consultan al momento de elegir una vivienda. No obstante, se procederá a
# analizar cuales de ellas efectivamente tienen relación con este objetivo y
# reconsiderar la elección en caso de ser necesario.
# %%
URL_MELB_HOUSING_DATA = "https://www.famaf.unc.edu.ar/~nocampo043/melb_housing_df.csv"
URL_MELB_SUBURB_DATA = "https://www.famaf.unc.edu.ar/~nocampo043/melb_suburb_df.csv"

prefix_labels = lambda prefix, labels: [
    prefix + "_" +  label for label in labels
]

categories = ["housing", "suburb"]
relevant_columns = {
    "housing" : [
        "price",
        "type",
        "selling_method",
        "seller_agency",
        "room_count",
        "bathroom_count",
        "garage_count",
        "building_area",
        "land_size",
        "cbd_distance",
        "year_built",
    ],
    "suburb": [
        "council_area",
        "name",
        "postcode",
        "region_name",
        "property_count",
        "council_area"
    ]
}

housing_columns = prefix_labels("housing", relevant_columns["housing"])
suburb_columns = prefix_labels("suburb", relevant_columns["suburb"])

melb_housing_df = pd.read_csv(
    URL_MELB_HOUSING_DATA,
    usecols=housing_columns
)
melb_suburb_df = pd.read_csv(
    URL_MELB_SUBURB_DATA,
    usecols=suburb_columns
)
# %%
melb_suburb_df
# %%
melb_housing_df
# %% [markdown]
# ## Datos erróneos
# Inicialmente se trabajó sobre el conjunto de datos sobre la identificación de
# datos erróneos. Recordemos que estos se caracterizan por expresar atipicidad o
# mal formación en los datos. Estos datos erróneos pueden verse reflejados en
# algúnas de las variables seleccionadas que podrían llevar a un valor atípico
# sobre el precio que se quiere estimar. Por ende, se realizó un análisis sobre
# cada una de ellas verificando si es relevante para estimar el precio y si hay
# inconsistencias que contradigan otras variables o conocimiento del dominio del
# problema.
# %% [markdown]
# ## Datos Erroneos: Valores atípicos
# Se consideró los valores atípicos correspondientes al precio de las
# propiedades, es decir, bajo la columna `housing_price`. Para ello se filtraron
# aquellos datos que se encuentren alejados de la media, más alla de 2.5 veces
# su desvío estandar.
# %%
melb_housing_df, melb_housing_outliers_df = clean_outliers(
    melb_housing_df,
    "housing_price"
)
# %%
melb_housing_df
# %%
melb_housing_outliers_df
# %% [markdown]
# Ahora bien, es importante analizar que entradas estan siendo removidas. Dado
# que durante el análisis se comparó si cada variable es relevante para estimar
# el precio de una propiedad, se trabajó sobre estos últimos dos *dataframes*
# para no solo visualizar sus columnas, si no también decidir si conservar o no
# los *outliers* influyen en la estimación.
# %% [markdown]
# ### Precio
# %%
melb_housing_df["housing_price"].describe().round(2)
# %%
plt.figure(figsize=(10,10))
seaborn.displot(melb_housing_df["housing_price"])
plt.ticklabel_format(style="plain", axis="x")
# %%
plt.figure(figsize=(8,8))
seaborn.boxenplot(data=melb_housing_df, x="housing_price")
plt.ticklabel_format(style="plain", axis="x")
# %% [markdown]
# ### Tipo de vivienda
# %%
melb_housing_df["housing_type"].unique()
# %% [markdown]
# Significado de cada fila:
# 
# - `h`: Casa.
# 
# - `u`: Unidad, dúplex.
# 
# - `t`: Casa adosada.
# %%
(
    melb_housing_df[["housing_type", "housing_price"]]
        .groupby("housing_type")
        .describe()
        .round(2)
)
# %%
fig = plt.figure(figsize=(8,6))
seaborn.barplot(
    data=melb_housing_df,
    x="housing_type",
    y="housing_price",
    estimator=np.mean
)
plt.ylabel("Media de Precio")
plt.xlabel("Type")
plt.ticklabel_format(style="plain", axis="y")
# %%
plt.figure(figsize=(8,8))
seaborn.boxenplot(
    data=melb_housing_df,
    x="housing_price",
    y="housing_type"
)
plt.ticklabel_format(style="plain", axis="x")
# %% [markdown]
# Observar que la variable tipo de la casa, tiene influencia en el precio de la
# propiedad.
#
# - Para el tipo `h`, es decir casa, el precio medio se encuentra en valores
#   cercanos a 1,2 millones. Estando el rango intercuartil comprendido entre los
#   780 mil y 1,4 millones. Se observa que los valores máximos son superiores a
#   los 2,5 millones.
#
# - Para el caso de `u`, es decir duplex, el precio medio se encuentra en valores
#   cercanos a los 600 mil. Estando el rango intercuartil comprendido entre los
#   400 mil y 700 mil. Se observa que los valores máximos son superiores a los
#   2,5 millones.
#
# - Para el caso de `t`, es decir casa adosada, el precio medio se encuentra en
#   valores un poco superiores a los 900 mil. Estando el rango intercuartil
#   comprendido entre los 670 mil y 1,1 millones. Se observa que los valores
#   máximos llegan a casi los 2,5 millones.
# %% [markdown]
# Visualizando los valores extremos del precio en función de esta variable se
# obtiene.
# %%
(
    melb_housing_outliers_df[["housing_type", "housing_price"]]
        .groupby("housing_type")
        .describe()
        .round(2)
)
# %%
plt.figure(figsize=(8,8))
seaborn.boxenplot(
    data=melb_housing_outliers_df,
    x="housing_price",
    y="housing_type"
)
plt.ticklabel_format(style="plain", axis="x")
# %% [markdown]
# Se observa que dentro de los outliers se están eliminando en mayor cantidad
# casas con precios comprendidos entre los 3 millones y los 9 millones.
# %% [markdown]
# ### Método de venta
# %%
melb_housing_df["housing_selling_method"].unique()
# %%
(
    melb_housing_df[["housing_selling_method", "housing_price"]]
        .groupby("housing_selling_method").describe().round(2)
)
# %%
plt.figure(figsize=(8,8))
seaborn.boxenplot(
    data=melb_housing_df,
    x="housing_price",
    y="housing_selling_method"
)
plt.ticklabel_format(style="plain", axis="x")
# %%
types = melb_housing_df["housing_type"].unique()
fig, axes = plt.subplots(3, figsize=(12,12))

for ax, type in zip(axes, types):
    houses_by_type_df = melb_housing_df[
        melb_housing_df["housing_type"] == type
    ]
    seaborn.boxenplot(
        ax=ax,
        data=houses_by_type_df,
        x="housing_price",
        y="housing_selling_method"
    )
    ax.ticklabel_format(style='plain', axis='x')
# %% [markdown]
# Observar que la distribución de la variable `housing_price` es similar en cada
# método de venta. Los valores medios están cercanos al millón extendiendose
# hasta valores máximos cercanos a los 2,5 millones. El método de venta `SA`,
# correspondiente a "vendido después de la subasta", parece ser el más
# diferente. No obstante, se observa que son pocos los casos comprendidos en
# esta categoria (menos de 100), por lo cual la baja frecuencia podría
# justificar su disparidad con el resto.
# %% [markdown]
# ### Agencia de ventas
# %%
melb_housing_df["housing_seller_agency"].value_counts()
# %% [markdown]
# Existen 266 vendedores que efectúan las transacciones de las viviendas. A
# continuación, se muestra si existe concentración de movimientos en algúno de
# ellos. 
# %%
best_sellers_df = (
    melb_housing_df[["housing_seller_agency", "housing_price"]]
        .groupby("housing_seller_agency")
        .agg(

            sales_count=("housing_seller_agency", "count"),
            sales_percentage=("housing_seller_agency", 
                lambda sales: 100 * len(sales) / len(melb_housing_df)
            )
        )
        .sort_values(by="sales_count", ascending=False)
        .head(20)
)
# %%
best_sellers_df.sum()
# %%
best_sellers_df.index
# %%
fig = plt.figure(figsize=(12,12))
seaborn.barplot(
    data=melb_housing_df[
        melb_housing_df["housing_seller_agency"].isin(
            best_sellers_df.index
        )
    ],
    x='housing_seller_agency',
    y='housing_price',
    estimator=np.mean
)
plt.xlabel("Agencia de ventas")
plt.ylabel("Precio Promedio de Ventas")
plt.xticks(rotation=90)
plt.ticklabel_format(style='plain', axis='y')
# %% [markdown]
# Se puede observar que algunos vendedores en promedio han vendido casas a
# precios más altos que otros, por ejemplo el vendedor `Marshall` sobresale por
# el resto con un precio medio de venta de 1,5 millones. Sin embargo, no se
# puede asegurar que el mayor precio de la venta sea por una mejor gestión del
# vendedor y no por otro tipo de variable, como ser el tipo de casa, la
# ubicación o bien su tamaño o composición.
# %% [markdown]
# ## 