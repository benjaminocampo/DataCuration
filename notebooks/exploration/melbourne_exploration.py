# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     cell_markers: '"""'
#     cell_metadata_filter: -all
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
"""
# Diplomatura en Ciencias de Datos, Aprendizaje Automático y sus Aplicaciones

Autores: Matías Oria, Antonela Sambuceti, Pamela Pairo, Benjamín Ocampo
"""
# %% [markdown]
"""
## Definición de constantes, funciones *helper*, y lectura del conjunto de datos

Se trabajó sobre los *dataframes* `melb_suburb_df` y `melb_housing_df` que
fueron obtenidos en `normalize_melbourne_dataset.ipynb` alojados en un servidor
de la Universidad Nacional de Córdoba para facilitar su acceso remoto.
"""
# %% [markdown]
"""
En caso de estar trabajando esta notebook desde Google Colab, se debe ejecutar
la siguiente celda para instalar el paquete `geopandas` ya que no se encuentra
disponible por defecto y es necesario para algunas visualizaciones de la
exploración. Si se está utilizando un entorno de Conda junto a Jupyter Notebook
con la configuración dada en `README.md` del repositorio, este paso no es
necesario.
"""
# %%
# !pip install geopandas
# %%
from typing import Tuple, List, Optional
import pandas as pd
import numpy as np
import seaborn
import matplotlib.pyplot as plt
import geopandas as gpd
import requests
import missingno as msno


def clean_outliers(df: pd.DataFrame,
                   column_name: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Filters out entries of @df that have in @column_name values which are 2.5
    times standard deviations apart from the mean. Returns both, entries that
    hold and miss the condition.
    """
    col = df[column_name]
    mask_outlier = np.abs(col - col.mean()) <= (2.5 * col.std())
    return df[mask_outlier], df[~mask_outlier]


def to_categorical(column: pd.Series, bin_size: int, min_cut: int,
                   max_cut: int) -> pd.Series:
    """
    Returns a pandas series where each value of @column is replaced by an
    interval that contains it. Intervals are generated from @min_cut to @max_cut
    and have size @bin_size.
    """
    if min_cut is None:
        min_cut = int(round(column.min())) - 1
    value_max = int(np.ceil(column.max()))
    max_cut = min(max_cut, value_max)
    intervals = [(x, x + bin_size) for x in range(min_cut, max_cut, bin_size)]
    if max_cut != value_max:
        intervals.append((max_cut, value_max))
    return pd.cut(column, pd.IntervalIndex.from_tuples(intervals))


def plot_melbourne_map(locations_df: gpd.GeoDataFrame,
                       key_regions: List[str],
                       column_name_colorbar: Optional[str] = None) -> None:
    """
    Plots a map of the surroundings of Melbourne, displaying the regions given
    by @key_regions. @location_df needs to be a geodataframe that contains the
    latitude and longitude as pandas points, so they can be superimposed on the
    background map. If @column_name_colorbar is provided, it needs to be the
    name of a column of @locations_df. Then, it colors the points and adds a
    colorbar indicating the magnitude of the values contained in that column.
    """
    geoserver = "https://data.gov.au/geoserver"
    route = "vic-state-electoral-boundaries-psma-administrative-boundaries"
    service = "wfs"
    projection = "EPSG:3110"
    wfsurl = f"{geoserver}/{route}/{service}"

    params = dict(service="WFS",
                  version="2.0.0",
                  request="GetFeature",
                  typeName=(route +
                            ":ckan_a0d8838b_2423_4c8b_a7d9_b04eb240a2b1"),
                  outputFormat="json")
    features = requests.get(wfsurl, params=params).json()
    region_location_df = gpd.GeoDataFrame.from_features(features).set_crs(
        projection)

    only_key_regions = region_location_df["vic_stat_2"].isin(key_regions)
    region_location_df = (
        region_location_df.loc[only_key_regions,
                               ["vic_stat_2", "geometry"]].dissolve(
                                   by="vic_stat_2").reset_index())
    background = region_location_df.plot(column="vic_stat_2",
                                         edgecolor="black",
                                         figsize=(15, 15),
                                         legend=True)
    plt.ylabel("Latitude")
    plt.xlabel("Longitude")
    background.set(title="Regiones del Territorio de Victoria, Australia")

    if column_name_colorbar is not None:
        cmap = seaborn.color_palette("flare", as_cmap=True)
        points = locations_df.plot(ax=background,
                                   marker="o",
                                   markersize=3,
                                   column=column_name_colorbar,
                                   cmap=cmap)

        fig = points.get_figure()
        cbax = fig.add_axes([0.95, 0.2, 0.04, 0.60])
        cbax.set_title(column_name_colorbar)
        sm = plt.cm.ScalarMappable(
            cmap=cmap,
            norm=plt.Normalize(vmin=min(locations_df[column_name_colorbar]),
                               vmax=max(locations_df[column_name_colorbar])))
        fig.colorbar(sm, cax=cbax, format="%d")
    else:
        points = locations_df.plot(ax=background,
                                   marker="o",
                                   markersize=3,
                                   color="r")
# %%
URL_MELB_HOUSING_DATA = "https://www.famaf.unc.edu.ar/~nocampo043/melb_housing_df.csv"
URL_MELB_SUBURB_DATA = "https://www.famaf.unc.edu.ar/~nocampo043/melb_suburb_df.csv"

melb_housing_df = pd.read_csv(URL_MELB_HOUSING_DATA)
melb_suburb_df = pd.read_csv(URL_MELB_SUBURB_DATA)
# %%
melb_suburb_df
# %%
melb_housing_df
# %% [markdown]
"""
## Elección de variables relevantes
Dado que el objetivo es predecir el precio de venta de viviendas en Melbourne se
procedió a analizar cuáles son las variables que influyen.
"""
# %% [markdown]
"""
### Precio de venta (`housing_price`)
"""
# %% [markdown]
"""
#### Eliminación de *outliers*
"""
# %%
plt.figure(figsize=(8, 8))
seaborn.boxenplot(data=melb_housing_df, x="housing_price")
plt.ticklabel_format(style="plain", axis="x")
# %% [markdown]
"""
Se observa la presencia de outliers en la variable `housing_price`, por encima
de los 6 millones. Se decidió eliminar aquellos valores atípicos que se
encuentran alejados de la media, más allá de 2.5 veces su desviación estandar ya
que solo es de interés conocer el precio de venta de aquellas viviendas
comercializadas con mayor frecuencia.
"""
# %%
melb_housing_df, melb_housing_outliers_df = clean_outliers(
    melb_housing_df, "housing_price")
# %%
melb_housing_df
# %%
melb_housing_outliers_df
# %% [markdown]
"""
A continuación, se analizaron las demás variables sin los *outliers* del precio
de venta.
"""
# %%
plt.figure(figsize=(8,8))
seaborn.boxenplot(data=melb_housing_df, x="housing_price")
plt.ticklabel_format(style="plain", axis="x")
# %% [markdown]
"""
### Dirección de las viviendas (`housing_address`)
Si bien `housing_address` permite conocer la ubicación exacta de una propiedad,
no brinda información sobre aspectos de calidad que motiven la variación del
precio de una vivienda en Melbourne. Es un dato único que primero debería ser
agrupado en zonas para argumentar algún análisis. Agrupamiento que ya es
brindado en el conjunto de datos por medio de otras variables como el nombre del
suburbio, departamento, o el código postal.
"""
# %%
melb_housing_df["housing_address"].value_counts()
# %% [markdown]
"""
### Cantidad habitaciones (`housing_bedroom_count`)
Dado que las variables `housing_bedroom_count`, `housing_room_count` están
fuertemente correlacionadas se optó por conservar la última de estas porque
`housing_bedroom_count` proveniene de otro *dataset*.
"""
# %%
melb_housing_df[
    ["housing_bedroom_count", "housing_room_count"]
].corr()
# %% [markdown]
"""
### Cantidad de ambientes (`housing_room_count`)
"""
# %%
melb_housing_df[["housing_room_count"]].value_counts()
# %% [markdown]
"""
Se puede observar que la cantidad de ambientes varían entre 1 a 4, siendo
valores menos frecuentes aquellas que tienen 5 o más. Por ende, se decide
agrupar esta categoría.
"""
# %%
# Explicitly create a copy after adding the column to avoid chained indexes
melb_housing_df = melb_housing_df.assign(housing_room_segment=to_categorical(
    melb_housing_df["housing_room_count"], bin_size=1, min_cut=None,
    max_cut=4))
# %%
plt.figure(figsize=(16, 8))
seaborn.boxplot(x="housing_room_segment",
                y="housing_price",
                data=melb_housing_df)
plt.ylabel("Precio de la vivienda")
plt.xlabel("Cantidad de ambientes")
plt.ticklabel_format(style='plain', axis='y')
# %% [markdown]
"""
Luego del análisis individual de la cantidad de ambientes, se puede observar que
existe un aumento de la mediana del precio de venta y su variabilidad.
"""
# %% [markdown]
"""
### Cantidad de baños (`housing_bathroom_count`)

La cantidad de baños de las viviendas vendidas se encuentran en su mayoría entre
1 y 3 siendo valores más atípicos las que superan este rango. Por otro lado, se
encuentran propiedades con una cantidad de 0 baños lo cual resulta peculiar
recordando que los tipos de hogares en venta son casas, dúplex, y casas
adosadas.
"""
# %%
melb_housing_df[["housing_bathroom_count"]].value_counts()
# %% [markdown]
"""
A continuación, se procedió a reemplazar estos valores por el más frecuente dado
que se considera que no puede haber una propiedad sin baño.
"""
# %%
min_bathroom_count = 1
lt_one_bathroom = melb_housing_df["housing_bathroom_count"] < min_bathroom_count
melb_housing_df.loc[lt_one_bathroom, "housing_bathroom_count"] = 1
# %% [markdown]
"""
Ahora bien, para aquellas viviendas que presenten entre 3 a más baños se
agruparán en una sola categoría con el fin de asegurar que los grupos 1, 2, y 3
o más baños, presenten una cantidad mínima de registros.
"""
# %%
melb_housing_df = melb_housing_df.assign(
    housing_bathroom_segment=to_categorical(
        melb_housing_df["housing_bathroom_count"],
        bin_size=1,
        min_cut=None,
        max_cut=2))
# %%
seaborn.catplot(data=melb_housing_df,
                y="housing_price",
                x="housing_bathroom_segment",
                height=4,
                aspect=2)
# %% [markdown]
"""
Se puede observar una disminución en el rango de precios medida que aumenta la
cantidad de baños. Si bien el precio máximo es similar, el mínimo aumenta para
cada categoría.
"""
# %% [markdown]
"""
### Cantidad de garages (`housing_garage_count`)
"""
# %%
melb_housing_df["housing_garage_count"].value_counts()
# %%
melb_housing_df = melb_housing_df.assign(housing_garage_segment=to_categorical(
    melb_housing_df["housing_garage_count"],
    bin_size=1,
    min_cut=None,
    max_cut=2))
# %%
melb_housing_df["housing_garage_segment"].unique()

# %%
plt.figure(figsize=(16, 8))
seaborn.boxplot(x="housing_garage_segment",
                y="housing_price",
                data=melb_housing_df)
plt.ylabel("Precio de la vivienda")
plt.xlabel("Cantidad de garages")
plt.ticklabel_format(style='plain', axis='y')
# %% [markdown]
"""
A excepción de las viviendas con un garage, el resto de categorías pareciera que
se comportan de manera similar ante la variable precio, por lo tanto se decidió
no seleccionar `housing_garage_count`. TODO: **corregir xticks**
"""
# %% [markdown]
"""
### Tamaño de terreno (`housing_land_size`)
"""
# %%
seaborn.pairplot(data=melb_housing_df.sample(2500),
                 y_vars="housing_price",
                 x_vars="housing_land_size",
                 aspect=2,
                 height=4)
# %% [markdown]
"""
Si bien el gráfico no muestra una relación entre el precio de venta y el tamaño
del terreno, se cree que puede ser importante en la predicción junto con el
resto de las variables.
"""
# %% [markdown]
"""
### Área de construcción (`housing_building_area`)
Se considera que esta variable es importante para predecir el precio, por ende
se procedió a imputar sus valores faltantes en una sección posterior.
"""
# %%
msno.bar(melb_housing_df[["housing_price", "housing_building_area"]],
         figsize=(12, 6),
         fontsize=12)
# %%
plt.figure(figsize=(16, 8))
seaborn.boxplot(x="housing_building_area",
                data=melb_housing_df)
plt.ticklabel_format(style="plain", axis="x")

# %% [markdown]
"""
Se puede observar la presencia de un valor extremo de 44515. Se decidió eliminar este valor ya que se aleja demasiado del rango de viviendas que se estuvo considerando.
"""

# %%
big_area = melb_housing_df[
    melb_housing_df['housing_building_area'] > 10000]
big_area

# %%
melb_housing_df = melb_housing_df.drop(big_area.index)

# %%
melb_housing_df[[ "housing_building_area"]].describe()

# %% [markdown]
"""
### Tipo de vivienda (`housing_type`)
Significado de cada categoría:

- `h`: Casa.

- `u`: Unidad, dúplex.

- `t`: Casa adosada.
"""
# %%
(
    melb_housing_df[["housing_type", "housing_price"]]
        .groupby("housing_type")
        .describe()
        .round(2)
)
# %%
plt.figure(figsize=(8,8))
seaborn.boxenplot(data=melb_housing_df, x="housing_price", y="housing_type")
plt.ticklabel_format(style="plain", axis="x")
# %% [markdown]
"""
Se observa que la variable tipo de vivienda, tiene influencia en el precio de la
propiedad.

- Para el tipo `h`, es decir casa, el precio medio se encuentra en valores
  cercanos a 1,2 millones. Estando el rango intercuartil comprendido entre los
  780 mil y 1,4 millones. Se observa que los valores máximos son superiores a
  los 2,5 millones.

- Para el caso de `u`, es decir duplex, el precio medio se encuentra en valores
  cercanos a los 600 mil. Estando el rango intercuartil comprendido entre los
  400 mil y 700 mil. Se observa que los valores máximos son superiores a los 2,5
  millones.

- Para el caso de `t`, es decir casa adosada, el precio medio se encuentra en
  valores un poco superiores a los 900 mil. Estando el rango intercuartil
  comprendido entre los 670 mil y 1,1 millones. Se observa que los valores
  máximos llegan también a casi 2,5 millones.
"""
# %% [markdown]
"""
### Método de venta (`housing_selling_method`)
Significado de cada método:
- `PI` - Propiedad transferida.
- `S`  - Propiedad vendida.
- `SA` - Vendido después de subasta.
- `SP` - Propiedad vendida antes.
- `VB` - Oferta del proveedor.
"""
# %%
(
    melb_housing_df[["housing_selling_method", "housing_price"]]
        .groupby("housing_selling_method")
        .describe()
        .round(2)
)
# %%
plt.figure(figsize=(8,8))
seaborn.boxenplot(data=melb_housing_df,
                  x="housing_price",
                  y="housing_selling_method")
plt.ticklabel_format(style="plain", axis="x")
# %%
types = melb_housing_df["housing_type"].unique()
fig, axes = plt.subplots(3, figsize=(12,12))

for ax, type in zip(axes, types):
    houses_by_type_df = melb_housing_df[
        melb_housing_df["housing_type"] == type
    ]
    seaborn.boxenplot(ax=ax,
                      data=houses_by_type_df,
                      x="housing_price",
                      y="housing_selling_method")
    ax.ticklabel_format(style="plain", axis="x")
# %% [markdown]
"""
Se observa que la distribución de la variable `housing_price` es similar en cada
método de venta. Los valores medios están cercanos al millón extendiéndose hasta
valores máximos cercanos a los 2,5 millones. El método de venta `SA`,
correspondiente a “vendido después de la subasta”, parece ser el más diferente.
No obstante, se observa que son pocos los casos comprendidos en esta categoría
(menos de 100), por lo cual la baja frecuencia podría justificar su disparidad
con el resto.

Consideramos no seleccionar el método venta para un siguiente análisis.
"""
# %% [markdown]
"""
### Agencia de ventas (`housing_seller_agency`)
"""
# %%
melb_housing_df["housing_seller_agency"].value_counts()
# %% [markdown]
"""
Existen 266 vendedores que efectúan las transacciones de las viviendas. A
continuación, se calcula si existe concentración de movimientos en alguno de
ellos.
"""
# %%
best_sellers_df = (
    melb_housing_df[["housing_seller_agency", "housing_price"]]
        .groupby("housing_seller_agency")
        .agg(sales_count=("housing_seller_agency", "count"),
             sales_percentage=("housing_seller_agency",
                                lambda sales: 100 * len(sales) / len(melb_housing_df)))
        .sort_values(by="sales_count", ascending=False)
        .head(20)
)
# %%
best_sellers_df.sum()
# %%
fig = plt.figure(figsize=(12, 12))
seaborn.barplot(
    data=melb_housing_df[melb_housing_df["housing_seller_agency"].isin(
        best_sellers_df.index)],
    x="housing_seller_agency",
    y="housing_price",
    estimator=np.mean)
plt.xlabel("Agencia de ventas")
plt.ylabel("Precio promedio de ventas")
plt.xticks(rotation=90)
plt.ticklabel_format(style="plain", axis="y")
# %% [markdown]
"""
Se puede observar que algunos vendedores en promedio han vendido casas a precios
más altos que otros, por ejemplo el vendedor `Marshall` sobresale por el resto
con un precio medio de venta de 1,5 millones. Sin embargo, no se puede asegurar
que el mayor precio de la venta sea por una mejor gestión del vendedor y no por
otro tipo de variable, como ser el tipo de casa, la ubicación o bien su tamaño o
composición. Por lo tanto tampoco se decidió seleccionarla.
"""
# %% [markdown]
"""
### Región y distancia al distrito central comercial (`suburb_region_name`, y `housing_cbd_distance`)
Para analizar las medidas de tendencia central del precio de las viviendas por
región se realizó un boxplot como se muestra a continuación.
"""
# %%
plt.figure(figsize=(16, 8))
seaborn.boxplot(x="suburb_region_name",
                y="housing_price",
                palette="Set2",
                data=melb_housing_df.join(melb_suburb_df, on="suburb_id"))
plt.xticks(rotation=40)
plt.ylabel("Precio de venta")
plt.xlabel("Región")
plt.ticklabel_format(style="plain", axis="y")
# %% [markdown]
"""
`Southern Metropolitan` es la región con la media más alta en el precio de
viviendas. `Northern Metropolitan`, `Western Motropolitan`, y `South-Eastern
Metropolitan` parecieran seguir un comportamiento similar. De la misma manera
ocurre con `Eastern Victoria`, `Northern Victoria`, y `Western Victoria`.
"""
# %% [markdown]
"""
La siguiente tabla muestra que `Southern Metropolitan` es la región en donde se
registró una mayor cantidad de ventas de viviendas (4377) a diferencia de
`Eastern Victoria`, `Northern Victoria` y `Western Victoria` que muestran menos
de
100.
"""
# %%
(
    melb_housing_df
        .join(melb_suburb_df, on="suburb_id")
        .loc[:, "suburb_region_name"]
        .value_counts()
)
# %% [markdown]
"""
#### Geolocalización de propiedades por región
El objetivo es ver la geolocalización de los datos en las diferentes regiones
del Territorio de Victoria, Australia. A continuación, se muestra una imagen
extraída de Wikipedia.

<img
src="https://github.com/benjaminocampo/DataCuration/blob/master/notebooks/graphs/melbourne_by_region.png?raw=1"
alt="melbourne by region">

Se utilizó el servicio de [wfs de geoserver](https://data.gov.au/geoserver)
donde se obtiene una representación geométrica de las regiones.
"""
# %%
geoserver = "https://data.gov.au/geoserver"
route = "vic-state-electoral-boundaries-psma-administrative-boundaries"
service = "wfs"
projection = "EPSG:3110"

wfsurl = f"{geoserver}/{route}/{service}"

params = dict(service="WFS",
              version="2.0.0",
              request="GetFeature",
              typeName=(route + ":ckan_a0d8838b_2423_4c8b_a7d9_b04eb240a2b1"),
              outputFormat="json")

features = requests.get(wfsurl, params=params).json()

region_location_df = gpd.GeoDataFrame.from_features(features).set_crs(
    projection)

region_location_df.head()
# %% [markdown]
"""
De este *dataframe* se obtiene información correspondiente a las divisiones
gubernamentales de Melbourne. En particular, la columna `vic_stat_2` es aquella
que contiene los nombres de regiones, y `geometry` su representación geométrica
limítrofe. La proyección sobre las zonas de Melbourne es dada por el método
`set_crs` que establece coordenadas arbitrarias del espacio en una ubicación
particular del planeta. Sin embargo, hay varias zonas que se muestran en los
alrededores de Melbourne, por ende, se filtran por aquellas que correspondan a
las que se tiene registro en el conjunto de datos inicial.
"""
# %%
key_regions = [
    region.upper() for region in melb_suburb_df["suburb_region_name"].unique()
]
# %% [markdown]
"""
De manera similar, se necesita convertir las coordenadas de las propiedades
vendidas en puntos geométricos de GeoPandas para ser graficados junto a las
zonas recolectadas. En particular, en este caso serán representados como un
objeto `POINT`. El *dataframe* subyacente es el siguiente:
"""
# %%
locations_df = gpd.GeoDataFrame(
    melb_housing_df[["housing_lattitude", "housing_longitude"]],
    geometry=gpd.points_from_xy(
        melb_housing_df["housing_longitude"],
        melb_housing_df["housing_lattitude"])).set_crs("EPSG:3110")

locations_df.head()
# %% [markdown]
"""
Finalmente, se procede a graficar las zonas limítrofes de Melbourne
superponiendo las ubicaciones de las viviendas. El mapa muestra que la mayoría
de las ventas (en color rojo) se concentran en la región Metropolitana
(`South-Eastern Metropolitan`, `Southern Metropolitan`, `Western Metropolitan` y
`Northern Metropolitan`).
"""
# %%
plot_melbourne_map(locations_df, key_regions)
# %% [markdown]
"""
Para observar con mayor detalle la zona metropolitana, se filtran las entradas
de `region_location_df` y se incluye en el mapa la variable
`housing_cbd_distance` que indica la distancia que una propiedad tiene al
distrito central comercial. Se muestra que las viviendas más cerca al centro
(valores de distancia cercanos a cero de color naranja claro), se encuentran en
`Southern Metropolitan` donde también se encuentra la ciudad de Melbourne.
"""
# %%
metropolitan_regions = [
    region_name for region_name in key_regions
    if region_name.endswith("METROPOLITAN")
]

plot_melbourne_map(locations_df.join(melb_housing_df["housing_cbd_distance"]),
                   metropolitan_regions, "housing_cbd_distance")
# %% [markdown]
"""
También, se realizó otro mapa incluyendo a la variable `housing_price`, el cual
muestra que los precios de vivienda más altos se localizan en las regiones de
`Southern Metropolitan` y `Estearn Metropolitan`.
"""
# %%
plot_melbourne_map(locations_df.join(melb_housing_df["housing_price"]),
                   metropolitan_regions, "housing_price")
# %% [markdown]
"""
Estas observaciones dejan en evidencia que la localización de las viviendas
puede influir en el precio de las mismas. En este sentido, se decide incluir la
variable `suburb_region_name` en futuros análisis. Respecto a la variable
`housing_cbd_distance`, su mapa correspondiente muestra que los valores cercanos
al centro se ubican en la región `Southern Metropolitan` y aumenta a medida que
se aleja del mismo y cambia de región. Por lo tanto, a partir de conocer la
región en la que se ubica una vivienda se puede inferir su valor de distancia y
por ende `housing_cbd_distance` ofrecería información redundante. Entonces se
decidió no incluirla en futuros análisis.
"""
# %% [markdown]
"""
Se puede ver que las regiones `Western Victoria`, `Eastern Victoria`, y
`Northern Victoria` poseen una baja frecuencia. Por lo tanto, se procedió a
agruparlos bajo una misma categoría, denominada `Victoria`.
"""
# %%
(
    melb_housing_df
        .join(melb_suburb_df["suburb_region_name"], on="suburb_id")
        .groupby("suburb_region_name")
        .size()
)
# %%
melb_suburb_df = melb_suburb_df.assign(
    suburb_region_segment=melb_suburb_df["suburb_region_name"].replace(
        {
            "Western Victoria": "Victoria",
            "Eastern Victoria": "Victoria",
            "Northern Victoria": "Victoria"
        }))
# %%
(
    melb_housing_df
        .join(melb_suburb_df["suburb_region_segment"], on="suburb_id")
        .groupby("suburb_region_segment")
        .size()
)
# %%
plt.figure(figsize=(8, 8))
seaborn.boxenplot(data=melb_housing_df.join(
    melb_suburb_df["suburb_region_segment"], on="suburb_id"),
                  x="suburb_region_segment",
                  y="housing_price")
plt.ticklabel_format(style="plain", axis="y")
plt.xticks(rotation=40)
# %% [markdown]
"""
### Cantidad de propiedades por suburbio (`suburb_property_count`)
"""
# %%
plot_melbourne_map(
    (
        locations_df
            .join(melb_housing_df["suburb_id"])
            .join(melb_suburb_df["suburb_property_count"], on="suburb_id")
    ),
    metropolitan_regions, "suburb_property_count")
# %% [markdown]
"""
Se puede visualizar en los mapas anteriormente expuestos, que los suburbios que
tienen mayor cantidad de propiedades, no necesariamente son los que mayores
precios de venta tienen. Por ejemplo, en la región `SOUTHERN METROPOLITAN` se
encuentran los precios más altos (color violeta más oscuro), sin embargo en esa
región suburb_property_count presenta una mayor variabilidad. 
"""
# %% [markdown]
"""
### Departamento gubernamental (`suburb_council_area`)
Analizando las medidas de tendencia central para las variables
`suburb_council_area` y `housing_price` se observa que algunos departamentos
tienen una única vivienda con precio y otros como `Boroondara` tiene más de 1000
viviendas. Esto muestra la disparidad en la cantidad de ventas registradas en
los diferentes departamentos. Dado que está variable brinda información similar
a otras ya disponibles en el *dataset* no se la consideró relevante para estimar
el precio, sin embargo, se la seleccionó para continuar con el resto de las
consignas del entregable.
"""
# %%
(
    melb_housing_df
        .join(melb_suburb_df, on="suburb_id")
        .loc[:, "suburb_council_area"]
        .value_counts()
)
# %% [markdown]
"""
### Fecha de venta (`housing_date_sold`)
El tiempo en el que se vendió una propiedad puede ser relevante si se consideran
variables como inflación o burbujas inmoboliarias durante el período de venta.
Dado que el conjunto de datos corresponden a ventas efectuadas durante los años
2016 y 2017, es importante saber como fluctuó el precio durante este intervalo.
Por ende, se trabajó sobre esta variable convirtiendo inicialmente los datos en
objetos `datetime`.
"""
# %%
melb_housing_df = melb_housing_df.assign(
    housing_date_sold_datetime=pd.to_datetime(
        melb_housing_df["housing_date_sold"]))
melb_housing_df["housing_date_sold_datetime"]
# %%
seaborn.lineplot(data=melb_housing_df,
                 x="housing_date_sold_datetime",
                 y="housing_price")
plt.ticklabel_format(style="plain", axis="y")
plt.xticks(rotation=45)
# %% [markdown]
"""
No se observa una tendencia entre la fecha y el precio de venta. Las propiedades
vendidas fluctuan entre los 800000 a 120000 con una alta variabilidad que se
obtiene probablemente a que se están considerando no solo los años y meses, sino
también el día de la venta, siendo esto quizás no tan relevante si se desea
identificar un período donde se realizaron ventas de un alto valor.
"""
# %%
melb_housing_df = melb_housing_df.assign(
    housing_date_sold_datetime= pd.to_datetime(melb_housing_df["housing_date_sold_datetime"].dt.strftime("%Y-%m")))

seaborn.lineplot(data=melb_housing_df,
                 x="housing_date_sold_datetime",
                 y="housing_price")
plt.ticklabel_format(style="plain", axis="y")
plt.xticks(rotation=45)
# %% [markdown]
"""
Si nos quedamos solamente con el mes y año de venta, se observa una menor
variabilidad pero aún así no se perciben una tendencia clara en las
fluctuaciones del precio.
"""
# %%
melb_housing_df = melb_housing_df.assign(
    housing_date_sold_datetime=melb_housing_df["housing_date_sold_datetime"].
    dt.strftime("%Y-%m"))

plt.figure(figsize=(10, 10))
seaborn.barplot(data=melb_housing_df,
                x="housing_date_sold_datetime",
                y="housing_price",
                estimator=np.mean)
plt.xticks(rotation=45)
# %% [markdown]
"""
Se puede observar que los precios de venta en promedio son similares en los
diferentes meses. Por lo tanto, se cree que debido al bajo rango de fechas que
posee este dataset esta variable no sería significativa para estimar el precio
de venta de las casas. 
"""
# %% [markdown]
"""
### Año de construcción (`housing_year_built`)
"""

# %%
seaborn.boxenplot(melb_housing_df["housing_year_built"])

# %% [markdown]
"""
Se observa que los años de construcción de las viviendas se distribuyen entre
los años 1800 y 2010, con excepción de una sola propiedad construída en el año
1200. Dado que el objetivo es predecir el precio de venta de las viviendas, se
consideró seleccionar el rango que abarca la mayor cantidad de ventas,
eliminando el valor de vivienda construída en el 1200 por considerar que
tiene una baja probabilidad de ocurrencia.
"""

# %%
old_atypical_house = melb_housing_df[
    melb_housing_df['housing_year_built'] < 1800]
old_atypical_house

# %%
melb_housing_df = melb_housing_df.drop(old_atypical_house.index)
# %%
seaborn.lineplot(data=melb_housing_df,
                 x="housing_year_built",
                 y='housing_price')

# %% [markdown]
"""
Se observa que las viviendas más antiguas tienen precios de venta más altos en
comparación con las propiedades más nuevas. El año de construcción junto a la
fecha de venta dan información acerca de la antiguedad de la propiedad. Debido a
que las ventas fueron realizadas en un lapso corto de tiempo (2 años), se
decidió seleccionar unicamente la variable `housing_year_built` ya que se podría
obtener la misma información.
"""

# %% [markdown]
"""
### Variables seleccionadas

En conclusión para continuar con el análisis, se procedió a seleccionar las
siguientes variables:

- Precio de venta (`housing_price`)
- Cantidad de ambientes (`housing_room_segment`)
- Cantidad de baños (`housing_bathroom_segment`)
- Tamaño del terreno (`housing_land_size`)
- Tamaño de la construcción (`housing_bulding_area`)
- Tipo de vivienda (`housing_type`)
- Año de construcción (`housing_year_built`)
- Región (`suburb_region_segment`)
- Departamento gubernamental (`suburb_council_area`)
- Nombre de región (`suburb_name`)
"""

# %% [markdown]
"""
## Imputación
"""

# %% [markdown]
"""
### Departamento gubernamental (`suburb_council_area`) 
"""

# %% [markdown]
"""
Recordemos que en la notebook `combine_airbnb_dataset.ipynb` se imputaron los
valores faltantes de la variable `suburb_council_area` en función a la columna
`suburb`. Sin embargo, quedaron 6 filas sin poder imputar y corresponden a los
siguientes suburbios.
"""
# %%
melb_suburb_df[melb_suburb_df["suburb_council_area"].isna()]
# %% [markdown]
"""
Para asignar estos valores faltantes se buscaron los departamentos
gubernamentales de tales suburbios a partir de una [fuente externa de códigos
postales de Melbourne ](https://github.com/matthewproctor/australianpostcodes).

- `Burnside`: Melton - Bacchus Marsh
- `Attwood`: Hume
- `Plumpton`: Melton - Bacchus Marsh
- `New Gisborne`: Macedon Ranges
- `Wallan`: Macedon Ranges
- `Monbulk`: Yarra Ranges
"""
# %%
new_councils = {
    "Burnside": "Melton - Bacchus Marsh",
    "Attwood": "Hume",
    "Plumpton": "Melton - Bacchus Marsh",
    "New Gisborne": "Macedon Ranges",
    "Wallan": "Macedon Ranges",
    "Monbulk": "Yarra Ranges"
}

missing_suburbs = melb_suburb_df["suburb_name"].isin(new_councils.keys())

filled_suburbs = (
    melb_suburb_df
        .loc[missing_suburbs, "suburb_name"]
        .apply(lambda suburb: [new_councils[suburb]])
)

melb_suburb_df.loc[missing_suburbs, "suburb_council_area"] = filled_suburbs
melb_suburb_df[missing_suburbs]
# %% [markdown]
"""
### Columnas Dataset Airnb (`suburb_rental_dailyprice`)
"""

# %%
msno.bar(melb_suburb_df, figsize=(12, 6), fontsize=12, color='steelblue')

# %%
melb_suburb_df["suburb_rental_dailyprice"].isna().sum()

# %% [markdown]
"""
Luego de efectuar la combinación con el *Dataset* de Airnb, nos quedaron 114
valores nulos en la columna `suburb_rental_daylyprice`. A continuación se
efectúa la imputación de dicha variable.
"""

# %%
melb_suburb_df["suburb_rental_dailyprice"].describe()

# %%
plt.figure(figsize=(8, 8))
seaborn.boxenplot(data=melb_suburb_df, x="suburb_rental_dailyprice")
plt.ticklabel_format(style="plain", axis="x")

# %% [markdown]
"""
Podemos ver que la distribución es bastante simetrica (la media y la mediana se
encuentran en valores cercanos), por lo cual se imputó esta variable por su
valor medio.
"""

# %%
melb_suburb_df["suburb_rental_dailyprice"] = (
    melb_suburb_df["suburb_rental_dailyprice"]
        .fillna(melb_suburb_df["suburb_rental_dailyprice"].mean())
)
melb_suburb_df.suburb_rental_dailyprice.isna().sum()


# %%
msno.bar(melb_suburb_df, figsize=(12, 6), fontsize=12, color='steelblue')

# %% [markdown]
"""
Se observa que ya no existen valores faltantes en la columna
`suburb_rental_dailyprice`. Para un análisis posterior, se cree que una
imputación del tipo KNN nos podría dar mayor información sobre esta variable.
"""

# %% [markdown]
"""
### Creación del conjunto de datos
A continuación, se procedió a remover las columnas no seleccionadas y guardarlo
en un archivo `.csv`
"""
# %%
selected_housing_columns = [
    "housing_price",
    "housing_room_segment",
    "housing_bathroom_segment",
    "housing_land_size",
    "housing_building_area",
    "housing_type",
    "housing_year_built",
    "suburb_id"
]
selected_suburb_columns = [
    "suburb_name",
    "suburb_region_segment",
    "suburb_council_area",
    "suburb_rental_dailyprice"
]

melb_housing_filtered_df = melb_housing_df[selected_housing_columns]

melb_suburb_filtered_df = melb_suburb_df[selected_suburb_columns]
# %%
melb_housing_filtered_df.to_csv("melb_housing_filtered_df.csv", index=False)
melb_suburb_filtered_df.to_csv("melb_suburb_filtered_df.csv", index=False)

# %%
