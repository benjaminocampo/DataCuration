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
# ---

# %% [markdown]
"""
# Diplomatura en Ciencias de Datos, Aprendizaje Automático y sus Aplicaciones

Autores: Matías Oria, Antonela Sambuceti, Pamela Pairo, Benjamín Ocampo
"""
# %% [markdown]
"""
## Introducción

Se trabajó sobre el conjunto de datos de [la competencia
Kaggle](https://www.kaggle.com/dansbecker/melbourne-housing-snapshot) para la
estimación de precios de ventas de propiedades en Melbourne, Australia.

Este fue producido por [DanB](https://www.kaggle.com/dansbecker) de datos
provenientes del sitio [Domain.com.au](https://www.domain.com.au/) y se accedió
a través de un servidor de la Universidad Nacional de Córdoba para facilitar su
acceso remoto.

Con el fin de agilizar la exploración, se realizó una etapa de preprocesamiento
separando información de viviendas y suburbios. A su vez, se agregó nueva
información proveniente de datos recolectados por [Tyler
Xie](https://www.kaggle.com/tylerx) a través de la página de AirBnB, disponibles
en [una
publicación]((https://www.kaggle.com/tylerx/melbourne-airbnb-open-data?select=cleansed_listings_dec18.csv))
en su perfil de Kaggle.
"""
# %% [markdown]
"""
## Definición de funciones y constantes *helper*
A continuación se encuentran las funciones y constantes que se utilizaron
durante el preprocesamiento.
"""
# %%
import pandas as pd
import missingno as msno
import numpy as np
from sklearn.neighbors import BallTree
from typing import Dict, List


def replace_columns(df: pd.DataFrame, new_columns: Dict[str, Dict[str, str]]) -> pd.DataFrame:
    """
    Renames the columns in @df according to @new_columns. Names to replace need
    to be organized in categories, as the example shows, so then the resulting
    columns will be assigned that category as a prefix.

    <category>: {
        <old_name_1> : <new_name_1>
        <old_name_2> : <new_name_2>
        ...
    }
    """
    new_col_names = {
        original_name: category + '_' + new_name
        for category, cols in new_columns.items()
        for original_name, new_name in cols.items()
    }
    return df.rename(columns=new_col_names)


def closest_locations(df_centers: pd.DataFrame, df_locations: pd.DataFrame,
                      k: int) -> np.array:
    """
    Returns a dataset with the index of the k locations
    in df_locations that are closest to each row in df_centers.
  
    Both datasets must have columns latitude and longitude.
    """
    df_centers_ = df_centers.copy()
    df_locations_ = df_locations.copy()

    for column in df_locations_[["latitude", "longitude"]]:
        df_locations_[f'{column}_rad'] = np.deg2rad(
            df_locations_[column].values)
        df_centers_[f'{column}_rad'] = np.deg2rad(df_centers_[column].values)

    ball = BallTree(df_locations_[["latitude_rad", "longitude_rad"]].values,
                    metric='haversine')

    _, indices = ball.query(df_centers_[["latitude_rad",
                                         "longitude_rad"]].values,
                            k=k)
    return indices

def concatenate_str_cols(text_batch: pd.DataFrame) -> pd.DataFrame:
    """
    For each row concatenates the columns of @text_batch and returns a new dataframe
    """
    result = text_batch[text_batch.columns[0]]
    for col in text_batch.columns[1:]:
        result += "\n" + text_batch[col]
    return result
# %% [markdown]
"""
## Renombrado de columnas
Debido a que se manipularon las columnas del conjunto de datos a través de
Python, se renombraron las columnas para que respeten los [estándares de
código](https://www.python.org/dev/peps/pep-0008/) del lenguaje de programación
y caractericen de mejor manera los datos almacenados. En particular, columnas
como `Car`, `Distance`, `Date`, y `Method` carecen de expresividad y no reflejan
los datos que están registrados en ellas. Por último, se representaron con
prefijos aquellas que relacionan propiedades de viviendas y suburbios.
"""
# %%
URL_DOMAIN_DATA = "https://cs.famaf.unc.edu.ar/~mteruel/datasets/diplodatos/melb_data.csv"

new_columns = {
    "suburb": {
        "Suburb": "name",
        "Propertycount": "property_count",
        "Regionname": "region_name",
        "Postcode": "postcode",
        "CouncilArea": "council_area"
    },
    "housing": {
        "Address": "address",
        "Price": "price",
        "Rooms": "room_count",
        "Date": "date_sold",
        "Distance": "cbd_distance",
        "Car": "garage_count",
        "Landsize": "land_size",
        "BuildingArea": "building_area",
        "Type": "type",
        "Bathroom": "bathroom_count",
        "Bedroom2": "bedroom_count",
        "Method": "selling_method",
        "YearBuilt": "year_built",
        "Lattitude": "lattitude",
        "Longtitude": "longitude",
        "SellerG": "seller_agency"
    }
}

melb_df = (pd
    .read_csv(URL_DOMAIN_DATA)
    .pipe(replace_columns, new_columns)
)

melb_df
# %% [markdown]
"""
## Separación del conjunto de datos
Se observa que los datos asociados a los suburbios se repiten por cada vivienda,
debido a que se encuentran almacenados en un único *dataframe*. Por ejemplo, las
filas asociadas a los departamentos del suburbio *Abbotsford* tendrán el mismo
`suburb_name`, `suburb_postcode`, `suburb_region_name`, `suburb_council_area`, y
`suburb_property_count` desaprovechando espacio de memoria y complejizando el
estudio de la estructura del conjunto de datos. Por ende, se puede [separar el
*dataframe*](https://en.wikipedia.org/wiki/Third_normal_form) en dos;
`melb_suburb_df` y `melb_housing_df`, mantentiendo una *foreign key* en
`melb_housing_df`. Esto permite unir ambas tablas por medio de la operación
`join` y obtener el *dataframe* original.

Para separar `melb_df` como se mencionó anteriormente, se obtienen las columnas
de ambas categorías por medio de sus prefijos `housing` y `suburb` y se filtran
para obtener dos *dataframes* distintos. Posteriormente, se remueven los
duplicados de la tabla asociada a los suburbios.
"""
# %%
housing_cols = [col for col in melb_df if col.startswith("housing")]
suburb_cols = [col for col in melb_df if col.startswith("suburb")]

melb_suburb_df = melb_df[suburb_cols].drop_duplicates()
melb_suburb_df
# %% [markdown]
"""
No obstante, se observa que al aplicar `drop_duplicates` para eliminar las filas
que tengan  entradas repetidas, se obtiene información duplicada en los
suburbios. Esto se debe a que la columna `suburb_council_area` contiene no solo
datos faltantes, sino que también para un suburbio, información distinta. Por
ejemplo, para el caso del suburbio `Alphington`:
"""
# %%
melb_suburb_df[melb_suburb_df["suburb_name"] == "Alphington"]
# %% [markdown]
"""
En este caso, las entradas para todas las columnas son las mismas salvo la de
`suburb_council_area`. De manera similar, esto ocurre con otros suburbios.
"""
# %%
(
    melb_suburb_df[["suburb_name", "suburb_council_area"]]
        .groupby("suburb_name")
        .size()
)
# %% [markdown]
"""
Ahora bien, las entradas distintas no se puede considerar que son
inconsistentes, ya que hay suburbios en Melbourne que dependen de dos
departamentos gubernamentales como es el caso de `Alphington`. Por ende, se
agruparon en listas todos los departamentos a los cuales un suburbio pertenece.
Si todas las entradas de un suburbio presentan valores nulos, será dejado como
faltante para ser imputado en la etapa de curación.
"""
# %%
councils_df = (
    melb_suburb_df[["suburb_name", "suburb_council_area"]]
        .groupby("suburb_name")
        .agg(lambda councils:
             np.nan
             if councils.count() == 0
             else list(councils.dropna()))
)
councils_df
# %%
melb_suburb_df = (
    melb_suburb_df
        .drop(columns="suburb_council_area")
        .drop_duplicates()
        .merge(councils_df, on="suburb_name")
)
melb_suburb_df
# %% [markdown]
"""
De esta forma se obtienen valores únicos para los suburbios, manteniendo todos
los departamentos a los cuales un suburbio pertenece en listas. También, puede
verse que al combinar los datos los índices fueron alterados obteniendo un total
de 314. Finalmente, estos índices se agregan al conjunto de datos de las
viviendas en aquellas posiciones donde se tenía un suburbio asociado siendo su
*foreign key*.
"""
# %%
melb_housing_df = (
    melb_df[housing_cols + ["suburb_name"]]
        .replace({suburb_name: suburb_id
                 for suburb_name, suburb_id
                 in zip(melb_suburb_df["suburb_name"], melb_suburb_df.index)})
        .rename(columns={"suburb_name": "suburb_id"})
)
# %%
melb_housing_df
# %%
melb_suburb_df
# %% [markdown]
"""
Con esto se pueden analizar características de los suburbios sin considerar las
viviendas. Es decir, la información contenida en las columnas
`suburb_council_area`, `suburb_property_count`, `suburb_region_name`,
`suburb_postcode` permite caracterizar mejor cada suburbio.
"""
# %% [markdown]
"""
## Combinación de conjuntos de datos
Con el fin de estimar con mayor precisión el valor de venta de una propiedad se
aumentó los datos actuales utilizando otro conjunto de datos obtenido por
publicaciones de la plataforma de AirBnB en Melbourne en el año
2018.

Este es un conjunto de datos de *scrapings* del sitio realizado por [Tyler
Xie](https://www.kaggle.com/tylerx), disponibles en [una
publicación]((https://www.kaggle.com/tylerx/melbourne-airbnb-open-data?select=cleansed_listings_dec18.csv))
en su perfil de Kaggle.

En particular, se agregó información del entorno de las viviendas. Es importante
conocer factores de calidad de dichas zonas, tales como la seguridad,
concurrencia, disponibilidad de actividades recreativas, entre otras. Por otro
lado, conocer cuál es el precio en el que se alquilan las viviendas de una zona
por lo general está relacionado con cierta garantía de algunos de esos factores.

Por ende, de las distintas columnas que se encuentran en el conjunto de datos de
AirBnB se utilizaron las siguientes:

- `zipcode`: El código postal es un buen descriptor que involucra un conjunto de
  suburbios que se encuentran cerca.

- `latitude` y `longitude`: Similar a `zipcode` también puede ser utilizado para
  determinar cercanía entre dos ubicaciones.

- `neighborhood_overview`: Obtener las palabras más frecuentes que se mencionan
  en las descripciones de los barrios más cercanos da mucha información sobre
  los factores de calidad.

- `price`, `weekly_price`, `monthly_price`: Obtener en promedio los precios de
  alquiler de las viviendas de un suburbio, pueden ser relevantes en la
  estimación del costo de una propiedad.
"""
# %%
URL_AIRBNB_DATA = "https://cs.famaf.unc.edu.ar/~mteruel/datasets/diplodatos/cleansed_listings_dec18.csv"

interesting_cols = [
    "zipcode",
    "neighborhood_overview",
    "price",
    "weekly_price",
    "monthly_price",
    "latitude",
    "longitude"
]

airbnb_df = pd.read_csv(URL_AIRBNB_DATA, usecols=interesting_cols)
airbnb_df["zipcode"] = pd.to_numeric(airbnb_df.zipcode, errors="coerce")
airbnb_df
# %% [markdown]
"""
Posteriormente, son de interés aquellos `zipcodes` que tengan una cantidad
mínima de registros. Por ende, se seleccionan aquellos que son superiores a la
mediana del conteo de registros (27).
"""
# %%
zipcode_count_df = airbnb_df["zipcode"].value_counts()
zipcode_count_df
# %%
airbnb_df = airbnb_df[airbnb_df["zipcode"].isin(
    zipcode_count_df.index[zipcode_count_df > zipcode_count_df.median()])]
airbnb_df
# %%
msno.bar(airbnb_df, figsize=(12, 6), fontsize=12, color='steelblue')
# %% [markdown]
"""
La cantidad de datos faltantes en las variables `weekly_price` y `monthly_price`
luego de quitar los códigos postales poco frecuentes es mayor al 80%. Luego le
sigue `neighborhood_overview` con alrededor del 40%. Si bien no se realizará una
curación de datos sobre el conjunto de AirBnB hasta luego de combinarlo con el
original, visualizar estos datos permite considerar si las variables elegidas
presentan una cantidad de muestras significativa y replantear su selección.
"""
# %%
msno.matrix(airbnb_df,figsize=(12, 6), fontsize=12, color=[0,0,0.2])
# %% [markdown]
"""
`weekly_price` y `monthly_price` presentan datos faltantes situados en la
categoría MNAR, es decir, a través de algúna perdida sistemática. Por lo tanto,
no son tenidas en cuenta debido a la poca cantidad de ejemplares. Por otro lado,
los datos faltantes de la columna `neighborhood_overview` parecen ser debido a
perdidas aleatorias.
"""
# %%
airbnb_df = (
    airbnb_df
        .drop(columns=["weekly_price", "monthly_price"])
        .dropna()
)

airbnb_df
# %%
msno.bar(airbnb_df,figsize=(12, 6), fontsize=12, color='steelblue')
# %% [markdown]
"""
Luego de eliminar los valores faltantes para realizar la grupación, se obtuvo un
*dataframe* con 13554 datos por cada columna. Se calculó el precio promedio de
renta por día de las viviendas agrupado por código postal.
"""
# %%
airbnb_by_zipcode_df = (
    airbnb_df.groupby("zipcode")
    .agg(suburb_rental_dailyprice=("price", "mean"))
    .reset_index()
    .rename(columns={"zipcode": "suburb_postcode"})
)
airbnb_by_zipcode_df
# %% [markdown]
"""
Cabe recalcar que `suburb_rental_dailyprice` corresponde a información del
suburbio de las viviendas que tienen asociado un código postal, por lo tanto los
datos deben combinarse con la tabla `melb_suburb_df`. Esta es otra de las
justificaciones por las cuales es conveniente normalizar relaciones de datos.
Por ende, se combinaron los *dataframes* `melb_suburb_df` y
`airbnb_by_zipcode_df` renombrando la columna `zipcode` de este último.
"""
# %%
melb_suburb_df = melb_suburb_df.merge(airbnb_by_zipcode_df,
                                      how='left',
                                      on="suburb_postcode")
melb_suburb_df
# %%
msno.bar(melb_suburb_df,figsize=(12, 6), fontsize=12, color='steelblue')
# %% [markdown]
"""
Notar ahora que los datos faltantes correspondientes luego de combinar
`suburb_rental_dailyprice` es debido a aquellos códigos postales que no
figuraban en el conjunto de datos de AirBnB.
"""
# %% [markdown]
"""
## Otras variables para combinación de datos
En la sección anterior se utilizó el código postal para combinar la variable de
precio de renta del *dataframe* de AirBnB. Sin embargo, columnas como el nombre
del suburbio, o combinación por coordenadas también se podrían haber elegido.
Con el fin de combinar las descripciones de 5 los barrios más cercanos de una
propiedad se utilizó esté último método mencionado.
"""
# %%
group_size = 5
col_to_join = 'neighborhood_overview'

airbnb_locations = airbnb_df[['latitude', 'longitude', col_to_join]]
sales_locations = (
    melb_housing_df[['housing_lattitude', 'housing_longitude']]
        .rename(columns={"housing_lattitude": 'latitude',
                         'housing_longitude': 'longitude'})
)
closest_indices = closest_locations(sales_locations,
                                    airbnb_locations,
                                    k=group_size)

descriptions = [
    (
        airbnb_locations
            .iloc[closest_indices[:, position]][[col_to_join]]
            .rename(columns={col_to_join: f"{col_to_join}_{position}"})
            .reset_index(drop=True)
    ) for position in range(group_size)
]

closest_airbnb_descriptions_df = pd.concat(descriptions, axis=1).fillna('')

melb_housing_df[f"housing_closest_{col_to_join}"] = concatenate_str_cols(
    closest_airbnb_descriptions_df)
# %%
melb_housing_df
# %% [markdown]
"""
Para finalizar, `melb_suburb_df` y `melb_housing_df` fueron puestos a
disposición en servidores de FaMAF para su futura exploración. Estos pueden
encontrarse en:

- [Datos de
  viviendas](https://www.famaf.unc.edu.ar/~nocampo043/melb_housing_df.csv)
- [Datos de
  suburbios](https://www.famaf.unc.edu.ar/~nocampo043/melb_suburb_df.csv)

En el directorio `exploration` se continúa a partir de este conjunto de datos
modificado para determinar que variables son relevantes en la estimación del
precio venta de una vivienda en Melbourne.
"""
# %%
