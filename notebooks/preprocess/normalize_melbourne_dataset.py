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
# Con el fin de agilizar la exploración, se realizó una etapa de preprocesamiento
# separando información de viviendas y suburbios. A su vez, agregando nueva
# información proveniente de datos recolectados por [Tyler
# Xie](https://www.kaggle.com/tylerx), disponibles en [una
# publicación]((https://www.kaggle.com/tylerx/melbourne-airbnb-open-data?select=cleansed_listings_dec18.csv))
# en su perfil de Kaggle.
# %% [markdown]
# ## Definición de funciones y constantes *helper*
# TODO: Explicar que hay en esta celda, agregar docstrings, y especificación de
# tipos.
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
# Python se renombraron los nombres asignados para que respeten los [estándares
# de código](https://www.python.org/dev/peps/pep-0008/) del lenguaje de
# programación y caractericen de mejor manera los datos almacenados. En
# particular, columnas como `Car`, `Distance`, `Date`, y `Method`, carecen de
# expresividad y no reflejan que está registrado por ellas. Otras solo fueron
# escritas en minúsculas y separadas por *underscores*. Por último, se
# representaron con prefijos aquellas que relacionan propiedades de viviendas y
# suburbios.
# %%

URL_DOMAIN_DATA = 'https://cs.famaf.unc.edu.ar/~mteruel/datasets/diplodatos/melb_data.csv'

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
# ## Normalización del conjunto de datos
# Notar que los datos asociados a los suburbios se repiten por cada vivienda,
# debido a que se encuentran almacenados en un único *dataframe*. Por ejemplo,
# las filas asociadas a los departamentos del suburbio *Abbotsford* tendrán el
# mismo `suburb_name`, `suburb_postcode`, `suburb_region_name`,
# `suburb_council_area`, y `suburb_property_count` desaprovechando espacio de
# memoria y complejizando el estudio de la estructura del conjunto de datos. Por
# ende se puede [normalizar](https://en.wikipedia.org/wiki/Third_normal_form) la
# tabla `melb_df` separandola en dos `melb_suburb_df` y `melb_housing_df`,
# mantentiendo una *foreign key* en `melb_housing_df`. Esto permite unir ambas
# tablas por medio de la operación `join` y obtener el *dataframe* original. Si
# bien aún con esta organización no se obtiene una tercera forma normal, debido
# a que las regiones en `melb_suburb_df` también pueden ser tratadas de manera
# independiente, normalizar aún más complejizaría la realización de consultas
# sobre el conjunto de datos.
#
# Para separar `melb_df` como se mencionó, se obtienen las columnas de ambas
# categorias por medio de sus prefijos `housing` y `suburb` y se filtran para
# obtener dos *dataframes* distintos. Posteriormente, se remueven los duplicados
# de la tabla asociada a los suburbios.
# %%
housing_cols = [col for col in melb_df if col.startswith("housing")]
suburb_cols = [col for col in melb_df if col.startswith("suburb")]

melb_suburb_df = melb_df[suburb_cols].drop_duplicates()
melb_suburb_df
# %% [markdown]
# Sin embargo, notar que al aplicar `drop_duplicates` para eliminar las filas
# que tengan como entradas repetidas las 5 columnas seleccionadas aún así se
# obtiene información repetida en los suburbios. Esto se debe a que la columna
# `suburb_council_area` tiene no solo datos faltantes, si no que también para un
# suburbio, información distinta. Por ejemplo para el caso del suburbio
# `Alphington`:
# %%
melb_suburb_df[melb_suburb_df["suburb_name"] == "Alphington"]
# %% [markdown]
# En este caso, las entradas para todas las columnas son las mismas salvo la de
# `suburb_council_area`. De manera similar, esto ocurre con otros suburbio.
# %%
(
    melb_suburb_df[["suburb_name", "suburb_council_area"]]
        .groupby("suburb_name")
        .size()
)
# %% [markdown]
# Ahora bien, las entradas distintas no se puede considerar que son
# inconsistentes ya que hay suburbios en Melbourne que dependen de dos
# departamentos gubernamentales como es el caso de `Alphington`. Se agruparán en
# listas todos los departamentos a los cuales un suburbio pertenece. Si todas
# las entradas de un suburbio presentan valores nulos, será dejado como faltante
# para ser imputado en la etapa de exploración.
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
# Notar ahora como si se obtienen valores únicos para los suburbios, manteniendo
# en listas de python todos los departamentos a los cuales un suburbio
# pertenece. También, puede verse que al combinar los datos los índices fueron
# alterados obteniendo un total de 314. Finalmente, estos índices se agregan al
# conjunto de datos de las viviendas en aquellos posiciones donde se tenía un
# suburbio asociado siendo su *foreign key*.
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
# Con esto se pueden analizar propiedades de los suburbios sin considerar las
# viviendas. En particular, algo interesante es que cada suburbio tiene asociado
# un único código postal para este conjunto de datos. Notar que esta
# caracteristica no es debido a un error introducido al eliminar los duplicados,
# ya que se hizo considerando `suburb_cols` que relacionan toda la información
# de los suburbios. Si ese hubiese sido el caso, se tendría que haber realizado
# algo similar que con `suburb_council_area`.
#
# Otro beneficio de esta representación es que permite que un código postal se
# repita para más de un suburbio. Este es el caso para la ciudad de Melbourne y
# puede verse en esta [lista de
# suburbios](https://en.wikipedia.org/wiki/List_of_Melbourne_suburbs).
# %%
melb_suburb_df[["suburb_name", "suburb_postcode"]].value_counts()
# %% [markdown]
# ## Combinación de conjuntos de datos
# Con el fin de estimar con mayor presición el valor de venta una propiedad se
# aumentarán los datos actuales utilizando otro conjunto de datos obtenido por
# publicaciones de la plataforma de AirBnB en Melbourne en el año
# 2018.
#
# Este es un conjunto de datos de scrapings del sitio realizado por [Tyler
# Xie](https://www.kaggle.com/tylerx), disponibles en [una
# publicación]((https://www.kaggle.com/tylerx/melbourne-airbnb-open-data?select=cleansed_listings_dec18.csv))
# en su perfil de Kaggle.
#
# En particular, se agregará información del entorno de las viviendas, o mejor
# dicho de los suburbios en donde se encuentran. Es importante conocer factores
# de calidad de dichas zonas, tales como la seguridad, concurrencia,
# disponibilidad de actividades recreativas, entre otras. Por otro lado, conocer
# cuál es el precio en el que se alquilan las viviendas de una zona por lo
# general está relacionado a cierta garantía de algunos de esos factores.
#
# Por ende, de las distintas columnas que se encuentran en el conjunto de datos
# de AirBnB se utilizaron:
# 
# - `zipcode`: El código postal es un buen descriptor que involucra un conjunto
#   de suburbios que se encuentran cerca.
#
# - `neighborhood_overview`, `transit`, y `description`: Obtener en promedio las
#   palabras más frecuentes que se mencionan en las descripciones de las
#   viviendas, suburbios, y concurrencia da mucha información sobre los factores
#   de calidad.
#
# - `price`, `weekly_price`, `monthly_price`: Suburbios con precios altos
#   precios de alquiler usualmente está relacionada a lugares que tienen un
#   cierto estándar de calidad. Obtener en promedio los precios de alquiler que
#   constituyen las viviendas de un suburbio claramente pueden ser relevantes en
#   la estimación del costo de una propiedad.

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
# Posteriormente, son de interés aquellos zipcodes que tengan una cantidad
# mínima de registros. TODO: Se eligió el umbral 50 al visualizar los datos,
# ¿Hay alguna forma de encontrarlo mejor?
# %%
zipcode_count_df = airbnb_df["zipcode"].value_counts()
zipcode_count_df
# %%
airbnb_df = airbnb_df[
    airbnb_df["zipcode"].isin(
        zipcode_count_df.index[zipcode_count_df > 50]
    )
]
airbnb_df
# %% [markdown]
# Ahora bien, notar que la cantidad de datos faltantes en las variables
# `weekly_price` y `monthly_price` luego de quitar los código postales poco
# frecuentes es mayor a 80% del conjunto de datos. Luego les siguen
# `neighborhood_overview` y `transit` con alrededor del 40%. `description` solo
# presenta una pequeña fracción de datos nulos. Si bien no se realizará una
# curación de datos sobre el conjunto de AirBnB hasta luego de combinarlo con el
# de Domain, visualizar estos datos permite considerar si las variables elegidas
# presentan una cantidad de muestras significativa y replantear su selección.
# %%
msno.bar(airbnb_df,figsize=(12, 6), fontsize=12, color='steelblue')
# %% [markdown]
# Notar como también, `weekly_price` y `monthly_price` corresponden a datos
# faltantes situados en la categoría MNAR, es decir, a través de alguna perdida
# sistemática siendo complicados de recuperar. Por otro lado, las columnas
# `neighborhood_overview` y `transit` muestran ser perdidas aleatorias. TODO:
# Explicar un poco mejor. Comentar que se podrían imputar las descripciones.
# %%
msno.matrix(airbnb_df,figsize=(12, 6), fontsize=12, color=[0,0,0.2])
# %% [markdown]
# Debido a que para agrupar por código postal en `airbnb_df` se necesita
# realizar una agregación sobre valores no nulos, las columnas `weekly_price` y
# `monthly_price` son desconsideradas debido a la poca cantidad de ejemplares.
# `neighborhood_overview` y `transit` se les removió los valores nulos
# lidiando con ellos en una etapa posterior del procesamiento una vez ya
# combinados los *dataframes*.
# %%
airbnb_df = airbnb_df \
    .drop(columns=["weekly_price", "monthly_price"]) \
    .dropna()

airbnb_df
# %%
msno.bar(airbnb_df,figsize=(12, 6), fontsize=12, color='steelblue')
# %% [markdown]
# Se obtuvo el siguiente conjunto de datos con 11636 datos por cada columna o
# variables de interés. Ahora bien, para cada columna es de interés otro tipo de
# información. Para este caso se decidió por obtener la frecuencia de palabras
# en el caso de las descripciones, y el precio promedio de renta por día de las
# viviendas agrupando por código postal.
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
