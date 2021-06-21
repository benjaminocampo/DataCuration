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
import geopandas as gpd
import requests

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
        "lattitude",
        "longitude"
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

housing_columns = prefix_labels("housing", relevant_columns["housing"]) + ["suburb_id"]
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
    ax.ticklabel_format(style="plain", axis="x")
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
    x="housing_seller_agency",
    y="housing_price",
    estimator=np.mean
)
plt.xlabel("Agencia de ventas")
plt.ylabel("Precio Promedio de Ventas")
plt.xticks(rotation=90)
plt.ticklabel_format(style="plain", axis="y")
# %% [markdown]
# Se puede observar que algunos vendedores en promedio han vendido casas a
# precios más altos que otros, por ejemplo el vendedor `Marshall` sobresale por
# el resto con un precio medio de venta de 1,5 millones. Sin embargo, no se
# puede asegurar que el mayor precio de la venta sea por una mejor gestión del
# vendedor y no por otro tipo de variable, como ser el tipo de casa, la
# ubicación o bien su tamaño o composición.
# %% [markdown]
# ### Región
# Para analizar las medidas de tendencia central del precio de las viviendas por
# región se realizó un boxplot como se muestra a continuación.
# %%
# %%
plt.figure(figsize=(16, 8))
seaborn.boxplot(
    x="suburb_region_name",
    y="housing_price",
    palette="Set2",
    data=melb_housing_df.join(melb_suburb_df, on="suburb_id")
)
plt.xticks(rotation=40)
plt.ylabel("Precio de la vivienda")
plt.xlabel("Región")
plt.ticklabel_format(style="plain", axis="y")
# %% [markdown]
#`Souther Metropolitan` es la región con la media mas alta en el precio de las
# viviendas y también no posee outliers. Luego, se observa un gran número de
# outliers para las regiones `Northern Metropolitan`, `Western Metropolitan`,
# `Eastern Metropolitan` y `SouthEastern Metropolitan`. `Western Victoria` es la
# región con una media y mediana mas baja en el precio de las viviendas.
#
# TODO: Los outliers ya fueron removidos, está bien decir que se siguen observando?
# %% [markdown]
# La siguiente tabla muestra que `Southern Metropolitan` es la región en donde se
# registró una mayor cantidad de ventas de viviendas (4377) a diferencia de
# `Eastern Victoria`, `Northern Victoria` y `Western Victoria` que muestran menos de
# 100.
# %%
(
    melb_housing_df
        .join(melb_suburb_df, on="suburb_id")
        .loc[:, "suburb_region_name"]
        .value_counts()
)
# %% [markdown]
# Analizando las medidas de tendencia central para las variables `suburb_name` y
# `housing_price` se observa que algunos suburbios tienen una única vivienda con
# precio y otros como `Boroondara` tiene mas de 1000 viviendas. Esto muestra la
# disparidad en la cantidad de viviendas en los diferentes suburbios. 
# TODO: Es el suburbio o el departamento? Se muestra una tabla de medidas
# descriptivas pero solo se habla del conteo, hace falta hacer describe?
# %%
(
    melb_housing_df
        .join(melb_suburb_df, on="suburb_id")
        .loc[:, "suburb_council_area"]
        .value_counts()
)
# %% [markdown]
# ## Geolocalización de propiedades por región
# El objetivo es ver la geolocalización de los datos en las diferentes regiones
# del Territorio de Victoria, Australia. Para  A continuación se muestra una
# imagen extraída de Wikipedia.
#
# <img src="../graphs/melbourne_by_region.png" alt="melbourne by region">
#
# Para lograr esto se utiliza el servicio de [wfs de
# geoserver](https://data.gov.au/geoserver) donde se obtienen una representación
# geométrica de las regiones
# %%
geoserver = "https://data.gov.au/geoserver"
route = "vic-state-electoral-boundaries-psma-administrative-boundaries"
service = "wfs"

wfsurl = f"{geoserver}/{route}/{service}"

params = dict(
    service="WFS",
    version="2.0.0",
    request="GetFeature",
    typeName=(
        route + ":ckan_a0d8838b_2423_4c8b_a7d9_b04eb240a2b1"
    ),
    outputFormat="json"
)

region_location_df = gpd.GeoDataFrame.from_features(
    requests.get(wfsurl, params=params).json()
).set_crs("EPSG:3110")

region_location_df.head()
# %% [markdown]
# De este *dataframe* se obtiene información correspondiente a las divisiones
# gubernamentales de melbourne. En particular, la columna `vic_stat_2` es
# aquella que contiene los nombres de regiones, y `geometry` su representación
# geométrica limítrofe. La proyección sobre las zonas de Melbourne es dada por
# el método `set_crs` que establece coordenadas arbitrarias del espacio en una
# ubicación particular del planeta. Sin embargo, hay varias zonas que se
# muestran en los alrededores de Melbourne, por ende, se filtran por aquellas
# que correspondan a las que se tiene registro en el conjunto de datos inicial.
# %%
key_regions = [
    region.upper()
    for region in
    melb_suburb_df["suburb_region_name"].unique()
]

region_location_df = region_location_df[["vic_stat_2", "geometry"]]
only_key_regions = region_location_df["vic_stat_2"].isin(key_regions)

region_location_df = (
    region_location_df[only_key_regions]
        .dissolve(by="vic_stat_2")
        .reset_index()
)
region_location_df
# %% [markdown]
# Similarmente, se necesita convertir las coordenadas de las propiedades
# vendidas en puntos geométricos de GeoPandas para ser graficados junto a las
# zonas recolectadas. En particular, en este caso serán representados como un
# objeto `POINT`. El *dataframe* subyacente es el siguiente:
# %%
house_location_cols = [
    "housing_cbd_distance",
    "housing_lattitude",
    "housing_longitude"
]
suburb_location_cols = [
    "suburb_region_name",
    "suburb_property_count"
]

house_location_df = (
    melb_housing_df[house_location_cols + ["housing_price", "suburb_id"]]
        .join(melb_suburb_df[suburb_location_cols], on="suburb_id")
)

house_location_df = gpd.GeoDataFrame(
    house_location_df,
    geometry=gpd.points_from_xy(
        house_location_df["housing_longitude"],
        house_location_df["housing_lattitude"]
    )
).set_crs("EPSG:3110")

house_location_df.head()
# %% [markdown]
# Finalmente, se ubican los polígonos que representan las zonas limítrofes de
# Melbourne superponiendo las ubicaciones de las viviendas, para obtener una
# ubicación geográfica de las propiedades. El mapa muestra que la mayoria de las
# ventas (en color rojo) se concentran en la región de Metropolitana
# (`South-Eastern Metropolitan`, `Southern Metropolitan`, `Western Metropolitan` y
# `Northern Metropolitan`).
# %%
background = region_location_df.plot(
    column="vic_stat_2",
    edgecolor="black",
    figsize=(15, 15),
    legend=True
)

properties = house_location_df.plot(
    ax=background,
    marker="o",
    color="red"
)
background.set(title="Regiones del Territorio de Victoria, Australia")
plt.ylabel("Latitude")
plt.xlabel("Longitude")
# %% [markdown]
# Para observar detalladamente la zona metropolitana se filtran las entradas de
# `region_location_df` y se incluye en el mapa la variable
# `housing_cbd_distance` que indica la distancia que una propiedad tiene del
# distrito central comercial, para visualizar la existencia de algún patrón
# espacial. Se muestra que las viviendas mas cerca al centro (valores de
# distancia cercanos a cero de color naranja claro), se encuentran en `Southern
# Metropolitan` donde también se encuentra la ciudad de Melbourne.
# %%
metropolitan_regions = [
    region_name
    for region_name in key_regions
    if region_name.endswith("METROPOLITAN")
]

only_metropolitan = (
    region_location_df["vic_stat_2"]
        .isin(metropolitan_regions)
)
region_location_df = region_location_df[only_metropolitan]
# %%
cmap= seaborn.color_palette("flare", as_cmap=True)

background = region_location_df.plot(
    column="vic_stat_2",
    edgecolor="black",
    figsize=(11, 11),
    legend=True
)
properties = house_location_df.plot(
    ax=background,
    marker="o",
    markersize=3,
    column="housing_cbd_distance",
    cmap=cmap
)

background.set(title="Área Metropolitana- Victoria, Australia")

plt.ylabel("Latitude")
plt.xlabel("Longitude")

fig = properties.get_figure()

cbax = fig.add_axes([0.95, 0.2, 0.04, 0.60])   
cbax.set_title("Distancia al centro")

sm = plt.cm.ScalarMappable(
    cmap=cmap,
    norm=plt.Normalize(
        vmin=min(house_location_df["housing_cbd_distance"]),
        vmax=max(house_location_df["housing_cbd_distance"])
    )
)
fig.colorbar(sm, cax=cbax, format="%d")
# %% [markdown]
# También, se realizó otro mapa incluyendo a la variable `housing_price`, el
# cual muestra que los precios de vivienda mas altos se localizan en las
# regiones de `Southern Metropolitan` y `Estearn Metropolitan`.
# TODO: Se repite el código del gráfico. Capaz se podría hacer una función.
# %%
cmap= seaborn.color_palette("flare", as_cmap=True)

background = region_location_df.plot(
    column="vic_stat_2",
    edgecolor="black",
    figsize=(11, 11),
    legend=True
)
properties = house_location_df.plot(
    ax=background,
    marker="o",
    markersize=3,
    column="housing_price",
    cmap=cmap
)

background.set(title="Área Metropolitana- Victoria, Australia")

plt.ylabel("Latitude")
plt.xlabel("Longitude")

fig = properties.get_figure()

cbax = fig.add_axes([0.95, 0.2, 0.04, 0.60])   
cbax.set_title("Precio de venta")

sm = plt.cm.ScalarMappable(
    cmap=cmap,
    norm=plt.Normalize(
        vmin=min(house_location_df["housing_price"]),
        vmax=max(house_location_df["housing_price"])
    )
)
fig.colorbar(sm, cax=cbax, format="%d")
# %% [mardown]
# Estas observarciones dejan en evidencia que la localización de las viviendas
# puede influir en el precio de las mismas. En este sentido, se decide incluir
# las variables `suburb_region_name` y `suburb_name` en futuros análisis.
# Respecto a la variable `housing_cbd_distance`, su mapa correspondiente muestra
# que los valores cercanos al centro se ubican en la región `Southern
# Metropolitan` y aumenta a medida que se aleja del mismo y cambia de región.
# Por lo tanto, a partir de conocer la región en la que se ubica una vivienda se
# puede inferir su valor de distancia y por ende `housing_cbd_distance`
# ofrecería información redundante y no es incluida en los futuros análisis.
# %% [markdown]
# ### Cantidad de baños
# Notar que la cantidad de baños de las viviendas vendidas se encuentran entre 1
# y 3 siendo valores más atípicos las que superan este rango. Por otro lado, se
# encuentran propiedades con una cantidad de 0 baños lo cual resulta peculiar
# recordando que los tipos de hogares en venta eran casas, duplex, y casas
# adosadas.
# %%
melb_housing_df[["housing_bathroom_count"]].value_counts()
# %% [markdown]
# Visualizando aquellas propiedades sin baños se logra ver por sus coordenadas
# de latitud y longitud que se encuentran realmente cercas entre sí, llegando a
# pensar que fueron dados por un error sistemático. Además, dado que no se
# encuentran otras irregularidades en otras columnas (salvo por la falta de
# entradas `housing_building_area`, y `housing_year_built`) no se considera que
# podrían ser descartadas. Por ende, se procedió a delimitar que la mínima cantidad
# de baños posibles sería de 1, cambiando estos valores en 0.
# TODO: ¿Se puede hacer algo con el warning de actualización en un slice?
# Capaz antes de hacer eliminar los outliers.
# %%
min_bathroom_count = 1
melb_housing_df.loc[
    melb_housing_df["housing_bathroom_count"] < min_bathroom_count,
    "housing_bathroom_count"
] = 1
# %% [markdown]
# Ahora bien, para aquellas viviendas que presenten entre 3 a más baños se
# agruparán en una sola categoría con el fin de asegurar que los grupos 1, 2, 3,
# y 3 o más baños, presenten una cantidad mínima de registros.
# TODO: Agrupar baños.
# %% [markdown]
# ### Cantidad de ambientes o habitaciones
# %%
melb_housing_df[["housing_room_count"]].value_counts()
# %% [markdown]
# De manera similar se puede ver que la cantidad de ambientes varían entre 1 a
# 5, siendo valores menos frecuentes aquellas que tienen 6 o más. Por ende, se
# decide agrupar esta categoría de manera similar que en el caso de los baños.
# Algo a notar, es que no se puede asegurar que `housing_room_count` considere
# `housing_bedroom_count` en su conteo. Por ejemplo, en el caso de la siguiente
# tabla puede verse que 500 viviendas tienen un total de 2 baños y 2 ambientes,
# siendo que las propiedades podrían tener una sala de estar, dormitorios, o una
# sala dedicada para la cocina. Otros casos menos frecuentes son los registros
# que presentan una mayor cantidad de baños que ambientes.
# TODO: Agrupar por ambientes similar a los baños.
# %%
pd.crosstab(
    melb_housing_df["housing_bathroom_count"],
    melb_housing_df["housing_room_count"]
)
# %% [markdown]
# ### Influencia en el precio de garajes, ambientes y baños
# Otra pregunta interesante es si tener una mayor cantidad de instalaciones
# influye en el precio de venta de una propiedad.
# TODO: Combinar en una sola figura que muestre los 3 gráficos.
# %%
plt.figure(figsize=(16, 8))
seaborn.boxplot(
    x="housing_room_count",
    y="housing_price",
    data=melb_housing_df
)
plt.xticks(rotation=40)
plt.ylabel("Precio de la vivienda")
plt.xlabel("Cantidad de ambientes")
plt.ticklabel_format(style='plain', axis='y')
# %%
plt.figure(figsize=(16, 8))
seaborn.boxplot(
    x="housing_bathroom_count",
    y="housing_price",
    data=melb_housing_df
)
plt.xticks(rotation=40)
plt.ylabel("Precio de la vivienda")
plt.xlabel("Cantidad de baños")
plt.ticklabel_format(style='plain', axis='y')
# %%
plt.figure(figsize=(16, 8))
seaborn.boxplot(
    x="housing_garage_count",
    y="housing_price",
    data=melb_housing_df
)
plt.xticks(rotation=40)
plt.ylabel("Precio de la vivienda")
plt.xlabel("Cantidad de garajes")
plt.ticklabel_format(style='plain', axis='y')
# %% [markdown]
# Se puede ver un aumento de la variabilidad del precio a medida que aumentan la
# cantidad de ambientes y baños, a diferencia de la disponibilidad de garajes.
# TODO: Explicar mejor esta parte, capaz se podría usar scatterplots en lugar de
# graficos de caja ya que no se menciona mucho sobre medidas descriptivas.