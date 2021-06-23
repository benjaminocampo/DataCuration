# %% [markdown]
# # Diplomatura en Ciencas de Datos, Aprendizaje Automático y sus Aplicaciones
#
# Autores: Matias Oria, Antonela Sambuceti, Pamela Pairo, Benjamín Ocampo
# %% [markdown]
# ## Definición de constantes, funciones *helper*, y lectura del conjunto de datos
#
# Se trabajó sobre los *dataframes* `melb_suburb_df` y `melb_housing_df` que
# fueron obtenidos en `normalize_melbourne_dataset.ipynb` alojados en un
# servidor de la Universidad Nacional de Córdoba para facilitar su acceso
# remoto.
# %%
import pandas as pd
import numpy as np
import seaborn
import matplotlib.pyplot as plt
import geopandas as gpd
import requests
import missingno as msno

def clean_outliers(df, column_name):
    col = df[column_name]
    mask_outlier = np.abs(col - col.mean()) <= (2.5 * col.std())
    return df[mask_outlier], df[~mask_outlier]
# %%
URL_MELB_HOUSING_DATA = "https://www.famaf.unc.edu.ar/~nocampo043/melb_housing_df.csv"
URL_MELB_SUBURB_DATA = "https://www.famaf.unc.edu.ar/~nocampo043/melb_suburb_df.csv"

melb_housing_df = pd.read_csv(
    URL_MELB_HOUSING_DATA
)
melb_suburb_df = pd.read_csv(
    URL_MELB_SUBURB_DATA
)
# %%
melb_suburb_df
# %%
melb_housing_df
# %% [markdown]
# ## Elección de variables relevantes
# Dado que el objetivo es predecir el precio de venta de viviendas en Melbourne
# se procedió a analizar cuales son las variables que influyen.
# %% [markdown]
# ### Precio de venta (`housing_price`)
# %%
plt.figure(figsize=(8,8))
seaborn.boxenplot(data=melb_housing_df, x="housing_price")
plt.ticklabel_format(style="plain", axis="x")
# %% [markdown]
# Se observa la presencia de outliers en la variable `housing_price`, por encima
# de los 6 millones. Se decidió eliminar aquellos valores atípicos que se
# encuentran alejados de la media, mas allá de 2.5 veces su desviación estandar.
# TODO: Explicar cual es la razón de que se descarten estos valores.
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
# A continuación se analizaron las demás variables sin los *outliers* del precio
# de venta.
# %%
plt.figure(figsize=(8,8))
seaborn.boxenplot(data=melb_housing_df, x="housing_price")
plt.ticklabel_format(style="plain", axis="x")
# %% [markdown]
# ### Dirección de las viviendas (`housing_address`)
# Si bien `housing_address` permite conocer la ubicación exacta de una
# propiedad, no brinda información sobre aspectos de calidad que motiven la
# variación del precio de una vivienda en Melbourne. Es un dato único que
# primero debería ser agrupado en zonas para argumentar algún análisis.
# Agrupamiento que ya es brindado en el conjunto de datos por medio de otras
# variables como el nombre del suburbio, departamento, o el código postal.
# %%
melb_housing_df["housing_address"].value_counts()
# %% [markdown]
# ### Cantidad habitaciones (`housing_bedroom_count`)
# Dado que las variables `housing_bedroom_count`, `housing_room_count` están
# fuertemente correlacionadas se optó por conservar la ultima de estas porque
# `housing_bedroom_count` proveniene de otro *dataset*.
# %%
melb_housing_df[
    ["housing_bedroom_count", "housing_room_count"]
].corr()
# %% [markdown]
# ### Cantidad de ambientes (`housing_room_count`)
# %%
melb_housing_df[["housing_room_count"]].value_counts()
# %% [markdown]
# Se puede observar que la cantidad de ambientes varían entre 1 a 4, siendo
# valores menos frecuentes aquellas que tienen 5 o más. Por ende, se decide
# agrupar esta categoría.
# %%
gt5_df = melb_housing_df["housing_room_count"].replace({
    1: "1",
    2: "2",
    3: "3",
    4: "4",
    5: "5 o más",
    6: "5 o más",
    7: "5 o más",
    8: "5 o más",
    10: "5 o más"
})

melb_housing_df.loc[:, "housing_room_categ"] = gt5_df
# %%
plt.figure(figsize=(16, 8))
seaborn.boxplot(
    x="housing_room_categ",
    y="housing_price",
    data=melb_housing_df.sort_values(by="housing_room_categ")
)
plt.xticks(rotation=40)
plt.ylabel("Precio de la vivienda")
plt.xlabel("Cantidad de ambientes")
plt.ticklabel_format(style='plain', axis='y')
# %% [markdown]
# Luego del analisis individual de la cantidad de ambientes, se puede observar
# que existe un aumento de la mediana del precio de venta y su variabilidad.
# %% [markdown]
# ### Cantidad de baños (`housing_bathroom`)
#
# La cantidad de baños de las viviendas vendidas se encuentran en su mayoría
# entre 1 y 3 siendo valores más atípicos las que superan este rango. Por otro
# lado, se encuentran propiedades con una cantidad de 0 baños lo cual resulta
# peculiar recordando que los tipos de hogares en venta son casas, duplex, y
# casas adosadas.
# %%
melb_housing_df[["housing_bathroom_count"]].value_counts()
# %% [markdown]
# A continuación, se procedió a reemplazar estos valores por el más frecuente
# dado que se considera que no puede haber una propiedad sin baño.
# %%
min_bathroom_count = 1
lt_one_bathroom = melb_housing_df["housing_bathroom_count"] < min_bathroom_count
melb_housing_df.loc[
    lt_one_bathroom,
    "housing_bathroom_count"
] = 1
# %% [markdown]
# Ahora bien, para aquellas viviendas que presenten entre 3 a más baños se
# agruparán en una sola categoría con el fin de asegurar que los grupos 1, 2,
# y 3 o más baños, presenten una cantidad mínima de registros.
# %%
gt_two_df = melb_housing_df["housing_bathroom_count"].replace({
    1: "1",
    2: "2",
    3: "3 o más",
    4: "3 o más",
    5: "3 o más",
    6: "3 o más",
    7: "3 o más",
    8: "3 o más"
})
melb_housing_df.loc[:, "housing_bathroom_categ"] = gt_two_df
# %%
seaborn.catplot(
    data=melb_housing_df,
    y="housing_price",
    x="housing_bathroom_categ",
    height=4, aspect=2
)
# %% [markdown]
# Se puede observar una disminución en el rango de precios medida que aumenta la
# cantidad de baños. Si bien el precio máximo es similar, el mínimo aumenta para
# cada categoría.
# %% [markdown]
# ### Cantidad de garages (`housing_garages_count`)
# %%
melb_housing_df["housing_garage_count"].value_counts()
# %%
gt_four_df = melb_housing_df["housing_garage_count"].replace({
    0: "0",
    1: "1",
    2: "2",
    3: "3",
    4: "4 o más",
    5: "4 o más",
    6: "4 o más",
    7: "4 o más",
    8: "4 o más",
    9: "4 o más",
    10: "4 o más"
})
melb_housing_df.loc[:, "housing_garage_categ"] = gt_four_df
# %%
plt.figure(figsize=(16, 8))
seaborn.boxplot(
    x="housing_garage_categ",
    y="housing_price",
    data=melb_housing_df.sort_values(by="housing_garage_categ")
)
plt.xticks(rotation=40)
plt.ylabel("Precio de la vivienda")
plt.xlabel("Cantidad de garages")
plt.ticklabel_format(style='plain', axis='y')
# %% [markdown]
# A excepción de las viviendas con un garage, el resto de categorias pareciera
# que se comportan de manera similar ante la variable precio, por lo tanto se
# decidió no seleccionar `housing_garage_count`.
# TODO: Verlo con Aldana. ¿Decidir sacarla? ¿Como tomar la decisión de descartarlo?
# %% [markdown]
# ### Tamaño de terreno (`housing_land_size`)
# %% 
seaborn.pairplot(
    data=melb_housing_df.sample(2500),
    y_vars="housing_price",
    x_vars="housing_land_size",
    aspect=2, height=4
)
# %% [markdown]
# TODO: Verlo con Aldana. ¿Decidir sacarla? ¿Como tomar la decisión de descartarlo?
# %% [markdown]
# ## Area de construcción (`housing_building_area`)
# Se considera que esta variable es importante para predecir el precio, por ende
# se procedió a imputar sus valores faltantes en una sección posterior.
# %%
msno.bar(
    melb_housing_df[["housing_price", "housing_building_area"]],
    figsize=(12, 6),
    fontsize=12,
)
# %% [markdown]
# ### Tipo de vivienda (`housing_type`)
# Significado de cada categoría:
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
plt.figure(figsize=(8,8))
seaborn.boxenplot(
    data=melb_housing_df,
    x="housing_price",
    y="housing_type"
)
plt.ticklabel_format(style="plain", axis="x")
# %% [markdown]
# Se observa que la variable tipo de vivienda, tiene influencia en el precio de la
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
#   máximos llegan también a casi 2,5 millones.
# %% [markdown]
# Visualizando los valores extremos del precio en función de esta variable se
# obtiene.
# TODO: Cuando eliminar outliers? que análisis se hacen para quitarlos?
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
# ### Método de venta (`housing_selling_method`)
# Significado de cada método:
# - `PI` - Propiedad transferida.
# - `S`  - Propiedad vendida.
# - `SA` - Vendido después de subasta.
# - `SP` - Propiedad vendida antes.
# - `VB` - Oferta del proveedor.
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
# Se observa que la distribución de la variable `housing_price` es similar en
# cada método de venta. Los valores medios están cercanos al millón
# extendiendose hasta valores máximos cercanos a los 2,5 millones. El método de
# venta `SA`, correspondiente a "vendido después de la subasta", parece ser el
# más diferente. No obstante, se observa que son pocos los casos comprendidos en
# esta categoria (menos de 100), por lo cual la baja frecuencia podría
# justificar su disparidad con el resto.
#
# Consideramos no seleccionar el método venta para un siguiente análisis.
# %% [markdown]
# ### Agencia de ventas (`housing_seller_agency`)
# %%
melb_housing_df["housing_seller_agency"].value_counts()
# %% [markdown]
# Existen 266 vendedores que efectúan las transacciones de las viviendas. A
# continuación, se calcula si existe concentración de movimientos en algúno de
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
plt.ylabel("Precio promedio de ventas")
plt.xticks(rotation=90)
plt.ticklabel_format(style="plain", axis="y")
# %% [markdown]
# Se puede observar que algunos vendedores en promedio han vendido casas a
# precios más altos que otros, por ejemplo el vendedor `Marshall` sobresale por
# el resto con un precio medio de venta de 1,5 millones. Sin embargo, no se
# puede asegurar que el mayor precio de la venta sea por una mejor gestión del
# vendedor y no por otro tipo de variable, como ser el tipo de casa, la
# ubicación o bien su tamaño o composición. Por lo tanto tampoco se decidió
# seleccionarla.
# %% [markdown]
# ### Región y distancia al ditrito central comercial (`housing_region_name`, y `housing_cbd_distance`)
# Para analizar las medidas de tendencia central del precio de las viviendas por
# región se realizó un boxplot como se muestra a continuación.
# %%
plt.figure(figsize=(16, 8))
seaborn.boxplot(
    x="suburb_region_name",
    y="housing_price",
    palette="Set2",
    data=melb_housing_df.join(melb_suburb_df, on="suburb_id")
)
plt.xticks(rotation=40)
plt.ylabel("Precio de venta")
plt.xlabel("Región")
plt.ticklabel_format(style="plain", axis="y")
# %% [markdown]
# `Southern Metropolitan` es la región con la media mas alta en el precio de
# viviendas. `Northern Metropolitan`, `Western Motropolitan`, y `South-Eastern
# Metropolitan` parecieran seguir un comportamiento similar. De la misma manera
# ocurre con `Eastern Victoria`, `Northern Victoria`, y `Western Victoria`.
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
# ## Geolocalización de propiedades por región
# El objetivo es ver la geolocalización de los datos en las diferentes regiones
# del Territorio de Victoria, Australia. A continuación, se muestra una
# imagen extraída de Wikipedia.
#
# <img src="../graphs/melbourne_by_region.png" alt="melbourne by region">
#
# Se utilizó el servicio de [wfs de geoserver](https://data.gov.au/geoserver)
# donde se obtiene una representación geométrica de las regiones.
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
# gubernamentales de Melbourne. En particular, la columna `vic_stat_2` es
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
# De manera similar, se necesita convertir las coordenadas de las propiedades
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
# Finalmente, se procede a graficar las zonas limítrofes de Melbourne
# superponiendo las ubicaciones de las viviendas. El mapa muestra que la mayoria
# de las ventas (en color rojo) se concentran en la región Metropolitana
# (`South-Eastern Metropolitan`, `Southern Metropolitan`, `Western Metropolitan`
# y `Northern Metropolitan`).
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
# Para observar con mayor detalle la zona metropolitana, se filtran las entradas
# de `region_location_df` y se incluye en el mapa la variable
# `housing_cbd_distance` que indica la distancia que una propiedad tiene al
# distrito central comercial. Se muestra que las viviendas mas cerca al centro
# (valores de distancia cercanos a cero de color naranja claro), se encuentran
# en `Southern Metropolitan` donde también se encuentra la ciudad de Melbourne.
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
# %% [markdown]
# Estas observarciones dejan en evidencia que la localización de las viviendas
# puede influir en el precio de las mismas. En este sentido, se decide incluir
# la variable `suburb_region_name` en futuros análisis. Respecto a la variable
# `housing_cbd_distance`, su mapa correspondiente muestra que los valores
# cercanos al centro se ubican en la región `Southern Metropolitan` y aumenta a
# medida que se aleja del mismo y cambia de región. Por lo tanto, a partir de
# conocer la región en la que se ubica una vivienda se puede inferir su valor de
# distancia y por ende `housing_cbd_distance` ofrecería información redundante.
# Entonces se decidió no incluirla en futuros análisis.
# %% [markdown]
# Se puede ver que las regiones `Western Victoria`, `Eastern Victoria`, y
# `Northern Victoria` poseen una baja frecuencia. Por lo tanto, se procedió a
# agruparlos bajo una misma categoría, denominada `Victoria`.
# %%
(
    melb_housing_df
        .join(melb_suburb_df["suburb_region_name"], on="suburb_id")
        .groupby("suburb_region_name")
        .size()
)
# %%
victorian_region_df = melb_suburb_df["suburb_region_name"].replace({
    "Western Victoria":"Victoria",
    "Eastern Victoria":"Victoria",
    "Northern Victoria":"Victoria"
})
melb_suburb_df.loc[:, "suburb_region_categ"] = victorian_region_df
# %%
(
    melb_housing_df
        .join(melb_suburb_df["suburb_region_categ"], on="suburb_id")
        .groupby("suburb_region_categ")
        .size()
)
# %%
plt.figure(figsize=(8,8))
seaborn.boxenplot(
    data=melb_housing_df.join(
        melb_suburb_df["suburb_region_categ"],
        on="suburb_id"
    ),
    x="suburb_region_categ",
    y="housing_price"
)
plt.ticklabel_format(style="plain", axis="y")
plt.xticks(rotation=40)
# %% [markdown]
# ## Departamento gubernamental (`housing_council_area`)
# Analizando las medidas de tendencia central para las variables
# `suburb_council_area` y `housing_price` se observa que algunos suburbios
# tienen una única vivienda con precio y otros como `Boroondara` tiene mas de
# 1000 viviendas. Esto muestra la disparidad en la cantidad de ventas
# registradas en los diferentes suburbios.
#
# TODO: Es el suburbio o el departamento? Se muestra una tabla de medidas
# descriptivas pero solo se habla del conteo, hace falta hacer describe?
# %%
(
    melb_housing_df
        .join(melb_suburb_df, on="suburb_id")
        .loc[:, "suburb_council_area"]
        .value_counts()
)