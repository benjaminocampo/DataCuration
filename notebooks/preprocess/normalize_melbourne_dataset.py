# %% [markdown]
# # Diplomatura en Ciencas de Datos, Aprendizaje Automático y sus Aplicaciones
#
# Autores: Matias Oria, Antonela Sambuceti, Pamela Pairo, Benjamín Ocampo
# %% [markdown]
# ## Introducción
#
# Se trabajó sobre el conjunto de datos de [la compentencia
# Kaggle](https://www.kaggle.com/dansbecker/melbourne-housing-snapshot) para la
# estimación de precios de ventas de propiedades en Melbourne, Australia.
#
# Este fue producido por [DanB](https://www.kaggle.com/dansbecker) de datos
# provenientes del sitio [Domain.com.au](https://www.domain.com.au/) y se
# accedió a través de un servidor de la Universidad Nacional de Córdoba para
# facilitar su acceso remoto.
#
# Con el fin de agilizar la exploración, se realizó una etapa de
# preprocesamiento separando información de viviendas y suburbios. A su vez, se
# agregó nueva información proveniente de datos recolectados por [Tyler
# Xie](https://www.kaggle.com/tylerx) a través de la página de AirBnB,
# disponibles en [una
# publicación]((https://www.kaggle.com/tylerx/melbourne-airbnb-open-data?select=cleansed_listings_dec18.csv))
# en su perfil de Kaggle.
# %% [markdown]
# ## Definición de funciones y constantes *helper*
# A continuación se encuentran las funciones y constantes que se utilizaron
# durante el preprocesamiento.
# TODO: Agregar docstrings, y especificación de tipos.
# %%
import pandas as pd
import nltk
import missingno as msno
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download("stopwords")
nltk.download('punkt')

stopwords = set(nltk.corpus.stopwords.words("english"))

def replace_columns(df, new_columns):
    new_col_names = {
        original_name: category + '_' + new_name
        for category, cols in new_columns.items()
        for original_name, new_name in cols.items()
    }
    return df.rename(columns=new_col_names)

def remove_unimportant_words(s):
    """
    Removes from the string @s all the stopwords, digits, and special chars
    """
    special_chars = "-.+,[@_!#$%^&*()<>?/\|}{~:]"
    digits = "0123456789"
    invalid_chars = special_chars + digits

    reduced_text = "".join(c for c in s if not c in invalid_chars)

    reduced_text = " ".join(
        w.lower() for w in word_tokenize(reduced_text)
        if not w.lower() in stopwords
    )
    return reduced_text

def frequent_words(text_batch, threshold):
    joined_descritions = " ".join(
        remove_unimportant_words(text) for text in text_batch
    )
    tokens = word_tokenize(joined_descritions)
    return nltk.FreqDist(tokens).most_common(threshold)

# %% [markdown]
# ## Renombrado de columnas
# Debido a que se manipularán las columnas del conjunto de datos a través de
# Python, se renombraron las columnas para que respeten los [estándares de
# código](https://www.python.org/dev/peps/pep-0008/) del lenguaje de
# programación y caractericen de mejor manera los datos almacenados. En
# particular, columnas como `Car`, `Distance`, `Date`, y `Method` carecen de
# expresividad y no reflejan los datos que están registrados en ellas. Por
# último, se representaron con prefijos aquellas que relacionan propiedades de
# viviendas
# y suburbios.
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

melb_df = pd \
    .read_csv(URL_DOMAIN_DATA) \
    .pipe(replace_columns, new_columns)

melb_df
# %% [markdown]
# ## Separación del conjunto de datos
# Se observa que los datos asociados a los suburbios se repiten por cada
# vivienda, debido a que se encuentran almacenados en un único *dataframe*. Por
# ejemplo, las filas asociadas a los departamentos del suburbio *Abbotsford*
# tendrán el mismo `suburb_name`, `suburb_postcode`, `suburb_region_name`,
# `suburb_council_area`, y `suburb_property_count` desaprovechando espacio de
# memoria y complejizando el estudio de la estructura del conjunto de datos. Por
# ende, se puede [separar el
# *dataframe*](https://en.wikipedia.org/wiki/Third_normal_form) en dos;
# `melb_suburb_df` y `melb_housing_df`, mantentiendo una *foreign key* en
# `melb_housing_df`. Esto permite unir ambas tablas por medio de la operación
# `join` y obtener el *dataframe* original.
#
# Para separar `melb_df` como se mencionó anteriormente, se obtienen las
# columnas de ambas categorias por medio de sus prefijos `housing` y `suburb` y
# se filtran para obtener dos *dataframes* distintos. Posteriormente, se
# remueven los duplicados de la tabla asociada a los suburbios.
# %%
housing_cols = [col for col in melb_df if col.startswith("housing")]
suburb_cols = [col for col in melb_df if col.startswith("suburb")]

melb_suburb_df = melb_df[suburb_cols].drop_duplicates()
melb_suburb_df
# %% [markdown]
# No obstante, se observa que al aplicar `drop_duplicates` para eliminar las
# filas que tengan  entradas repetidas, se obtiene información duplicada en los
# suburbios. Esto se debe a que la columna `suburb_council_area` contiene no solo
# datos faltantes, sino que también para un suburbio, información distinta. Por
# ejemplo, para el caso del suburbio `Alphington`:
# %%
melb_suburb_df[melb_suburb_df["suburb_name"] == "Alphington"]
# %% [markdown]
# En este caso, las entradas para todas las columnas son las mismas salvo la de
# `suburb_council_area`. De manera similar, esto ocurre con otros suburbios.
# %%
(
    melb_suburb_df[["suburb_name", "suburb_council_area"]]
        .groupby("suburb_name")
        .size()
)
# %% [markdown]
# Ahora bien, las entradas distintas no se puede considerar que son
# inconsistentes ya que hay suburbios en Melbourne que dependen de dos
# departamentos gubernamentales como es el caso de `Alphington`. Por ende, se
# agruparán en listas todos los departamentos a los cuales un suburbio
# pertenece. Si todas las entradas de un suburbio presentan valores nulos, será
# dejado como faltante para ser imputado en la etapa de curación.
# %%
councils_df = (
    melb_suburb_df[["suburb_name", "suburb_council_area"]]
        .groupby("suburb_name")
        .agg(lambda councils:
            np.nan 
            if councils.count() == 0
            else list(councils.dropna())
        )
)
# %%
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
# TODO: Se puede hacer de otra forma? Evitar obtener listas de Python.
#
# De esta forma se obtienen valores únicos para los suburbios, manteniendo todos
# los departamentos a los cuales un suburbio pertenece en listas. También, puede
# verse que al combinar los datos los índices fueron alterados obteniendo un
# total de 314. Finalmente, estos índices se agregan al conjunto de datos de las
# viviendas en aquellas posiciones donde se tenía un suburbio asociado siendo su
# *foreign key*.
# %%
melb_housing_df = melb_df[housing_cols + ["suburb_name"]] \
    .replace({
        suburb_name: suburb_id
        for suburb_name, suburb_id
        in zip(melb_suburb_df["suburb_name"], melb_suburb_df.index)
    }) \
    .rename(columns={"suburb_name": "suburb_id"})
# %%
melb_housing_df
# %%
melb_suburb_df
# %% [markdown]
# Con esto se pueden analizar caracteristicas de los suburbios sin considerar
# las viviendas. Es decir, la información contenida en las columnas
# `suburb_council_area`, `suburb_property_count`, `suburb_region_name`,
# `suburb_postcode` permite caracterizar mejor cada suburbio.
# %% [markdown]
# ## Combinación de conjuntos de datos
# Con el fin de estimar con mayor precisión el valor de venta de una propiedad
# se aumentó los datos actuales utilizando otro conjunto de datos obtenido
# por publicaciones de la plataforma de AirBnB en Melbourne en el año
# 2018.
#
# Este es un conjunto de datos de *scrapings* del sitio realizado por [Tyler
# Xie](https://www.kaggle.com/tylerx), disponibles en [una
# publicación]((https://www.kaggle.com/tylerx/melbourne-airbnb-open-data?select=cleansed_listings_dec18.csv))
# en su perfil de Kaggle.
#
# En particular, se agregó información del entorno de las viviendas, o mejor
# dicho de los suburbios en donde se encuentran. Es importante conocer factores
# de calidad de dichas zonas, tales como la seguridad, concurrencia,
# disponibilidad de actividades recreativas, entre otras. Por otro lado, conocer
# cuál es el precio en el que se alquilan las viviendas de una zona por lo
# general está relacionado a cierta garantía de algunos de esos factores.
#
# Por ende, de las distintas columnas que se encuentran en el conjunto de datos
# de AirBnB se utilizaron las siguientes:
# 
# - `zipcode`: El código postal es un buen descriptor que involucra un conjunto
#   de suburbios que se encuentran cerca.
#
# - `neighborhood_overview`, `transit`, y `description`: Obtener las palabras
#   más frecuentes que se mencionan en las descripciones de las viviendas,
#   suburbios, y concurrencias, da mucha información sobre los factores de
#   calidad.
#
# - `price`, `weekly_price`, `monthly_price`: Obtener en promedio los precios de
#   alquiler de las viviendas de un suburbio, pueden ser relevantes en la
#   estimación del costo de una propiedad.

# %%

URL_AIRBNB_DATA = "https://cs.famaf.unc.edu.ar/~mteruel/datasets/diplodatos/cleansed_listings_dec18.csv"

interesting_cols = [
    "zipcode",
    "neighborhood_overview", "transit", "description",
    "price", "weekly_price", "monthly_price",
]

airbnb_df = pd.read_csv(URL_AIRBNB_DATA, usecols=interesting_cols)
airbnb_df["zipcode"] = pd.to_numeric(airbnb_df.zipcode, errors="coerce")
airbnb_df
# %% [markdown]
# Posteriormente, son de interés aquellos `zipcodes` que tengan una cantidad
# mínima de registros. Por ende, se seleccionan aquellos que son superiores a la
# mediana del conteo de registros (27).
# %%
zipcode_count_df = airbnb_df["zipcode"].value_counts()
zipcode_count_df
# %%
airbnb_df = airbnb_df[
    airbnb_df["zipcode"].isin(
        zipcode_count_df.index[zipcode_count_df > zipcode_count_df.median()]
    )
]
airbnb_df
# %% [markdown]
# Ahora bien, la cantidad de datos faltantes en las variables `weekly_price` y
# `monthly_price` luego de quitar los códigos postales poco frecuentes es mayor
# al 80%. Luego les siguen `neighborhood_overview` y `transit` con alrededor del
# 40%. `description` solo presenta una pequeña fracción de datos nulos. Si bien
# no se realizará una curación de datos sobre el conjunto de AirBnB hasta luego
# de combinarlo con el original, visualizar estos datos permite considerar si
# las variables elegidas presentan una cantidad de muestras significativa y
# replantear su selección.
# %%
msno.bar(airbnb_df,figsize=(12, 6), fontsize=12, color='steelblue')
# %% [markdown]
# `weekly_price` y `monthly_price` presentan datos faltantes situados en la
# categoría MNAR, es decir, a través de alguna perdida sistemática. Por lo
# tanto, son no tenidas en cuenta debido a la poca cantidad de ejemplares. Por
# otro lado, las clumnas `neighborhood_overview` y `transit` parecen ser
# perdidas aleatorias.
# %%
msno.matrix(airbnb_df,figsize=(12, 6), fontsize=12, color=[0,0,0.2])
# %%
airbnb_df = airbnb_df \
    .drop(columns=["weekly_price", "monthly_price"]) \
    .dropna()

airbnb_df
# %%
msno.bar(airbnb_df,figsize=(12, 6), fontsize=12, color='steelblue')
# %% [markdown]
# Se obtuvo el siguiente conjunto con 12395 datos por cada columna. Se calculó
# la frecuencia de palabras para cada descripción, y el precio
# promedio de renta por día de las viviendas agrupado por código postal.
# %%
ten_most_freq_words = lambda text_batch: frequent_words(text_batch, 10)

airbnb_df = airbnb_df.groupby("zipcode") \
    .agg(
        suburb_description_wordcount=(
            "description", ten_most_freq_words
        ),
        suburb_neighborhoods_overview_wordcount=(
            "neighborhood_overview", ten_most_freq_words
        ),
        suburb_transit_wordcount=("transit", ten_most_freq_words),
        suburb_rental_dailyprice=("price", "mean")
    ) \
    .reset_index() \
    .rename(columns={"zipcode": "suburb_postcode"})
# %%
airbnb_df
# %% [markdown]
# Cabe recalcar nuevamente que solo se incluirá información del suburbio de las
# viviendas por lo tanto los datos deben combinarse con la tabla
# `melb_suburb_df`. Esta es otra de las justificaciones por las cuales es
# conveniente normalizar relaciones de datos. Por ende, para finalizar se
# combinaron los *dataframes* `melb_suburb_df` y `airbnb_df` renombrando la
# columna `zipcode` de este último.
# %%
melb_suburb_df = melb_suburb_df.merge(
    airbnb_df, how='left', on="suburb_postcode"
)
# %%
melb_suburb_df
# %%
msno.bar(melb_suburb_df,figsize=(12, 6), fontsize=12, color='steelblue')
# %% [markdown]
# Notar ahora que los datos faltantes correspondientes luego de combinar los
# datos son debido a aquellos código postales que no figuraban en el conjunto de
# datos de AirBnB.
#
# Para finalizar, `melb_suburb_df` y `melb_housing_df` fueron puestos a
# disposición en servidores de FaMAF para su futura exploración. Estos pueden
# encontrarse en:
# - [Datos de
#   viviendas](https://www.famaf.unc.edu.ar/~nocampo043/melb_housing_df.csv)
# - [Datos de
#   suburbios](https://www.famaf.unc.edu.ar/~nocampo043/melb_suburb_df.csv)
# %%
