# %% [markdown]
# # Diplomatura en Ciencas de Datos, Aprendizaje Automático y sus Aplicaciones
#
# Autores: Matias Oria, Antonela Sambuceti, Pamela Pairo, Benjamín Ocampo %%
# %% [markdown]
# Se trabajó sobre el conjunto de datos de [la compentencia
# Kaggle](https://www.kaggle.com/dansbecker/melbourne-housing-snapshot) para la
# estimación de precios de ventas de propiedades en Melbourne, Australia.
#
# Este fue producido por [DanB](https://www.kaggle.com/dansbecker) de datos
# provenientes del sitio [Domain.com.au](https://www.domain.com.au/) y se
# accedió a través de un servidor de la Universidad Nacional de Córdoba para
# facilitar su acceso remoto.
#
# Posteriormente se realizó la exploración y curación de los datos que lo
# conforman centrandose en variables relevantes que favorezcan la estimación
# objetivo.
# 
# La exploración fue realizada principalmente por medio de
# [Pandas](https://pandas.pydata.org/) y librerias de manipulación de datos.
# %% [markdown]
# ## Definición de funciones y constantes *helper*
# %%
import pandas as pd
import nltk
from pandas.io.pytables import attribute_conflict_doc
import missingno as msno
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

def frequent_words(text_batch):
    joined_descritions = " ".join(
        remove_unimportant_words(text) for text in text_batch
    )
    tokens = word_tokenize(joined_descritions)
    return nltk.FreqDist(tokens)

def min_central_tendency(df, col, max_threshold):
    tendency = [
        (
            threshold,
            df[df[col] > threshold][col].mean(),
            df[df[col] > threshold][col].median()
        )
        for threshold in range(int(df[col].min()), max_threshold)
    ]

    tendency_df = pd.DataFrame(tendency, columns=['threshold', 'mean', 'median'])
    tendency_df["distance"] = abs(tendency_df["mean"] - tendency_df["median"])
    best_threshold = tendency_df.idxmin()["distance"]

    return (
        tendency_df.melt(id_vars='threshold', var_name='metric'),
        best_threshold
    )

# %% [markdown]
# ## Renombrado de columnas
# Debido a que se manipularán las columnas del conjunto de datos a través de
# Python se renombraron los nombres asignados para que respeten los [estándares
# de código](https://www.python.org/dev/peps/pep-0008/) del lenguaje de
# programación y caractericen de mejor manera los datos almacenados. En
# particular, columnas como `Car`, `Distance`, `Date`, y `Method`, carecen de
# expresividad y no reflejan que está registrado por ellas. Otras solo fueron
# escritas en minúsculas y separadas por *underscores*. Por último se
# representaron con prefijos aquellas que relacionan propiedades de viviendas y
# suburbios.
# %%

URL_DOMAIN_DATA = 'https://cs.famaf.unc.edu.ar/~mteruel/datasets/diplodatos/melb_data.csv'

new_columns = {
    "suburb": {
        "Suburb": "name",
        "Propertycount": "property_count",
        "Regionname": "region_name",
        "Postcode": "postcode"
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
        "CouncilArea": "council_area",
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
# mismo `suburb_name`, `suburb_postcode`, `suburb_region_name`, y
# `suburb_property_count` desaprovechando espacio de memoria y complejizando el
# estudio de la estructura del conjunto de datos. Por ende se puede
# [normalizar](https://en.wikipedia.org/wiki/Third_normal_form) la tabla
# `melb_df` separandola en dos `melb_suburb_df` y `melb_housing_df`,
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
# de la tabla asociada a los suburbios y les agregamos nuevos indices.
# Finalmente, se agregan esos indices al *dataframe* de las viviendas acorde a
# como se encontraban en el original.
# %%
housing_cols = [col for col in melb_df if col.startswith("housing")]
suburb_cols = [col for col in melb_df if col.startswith("suburb")]

melb_suburb_df = melb_df[suburb_cols].drop_duplicates().reset_index(drop=True)
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
# de los suburbios.
#
# Otro beneficio de esta representación es que permite que un código postal se
# repita para más de un suburbio. Este es el caso para la ciudad de Melbourne y
# puede verse en esta [lista de
# suburbios](https://en.wikipedia.org/wiki/List_of_Melbourne_suburbs).
# %%
melb_suburb_df[["suburb_name", "suburb_postcode"]].value_counts()
# %% [markdown]
# ## Combinación de conjunto de datos
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
# `neighborhood_overview` y `transit` muestran se perdidas aleatorias. TODO:
# Explicar un poco mejor. Comentar que se podrían imputar las descripciones.
# %%
msno.matrix(airbnb_df,figsize=(12, 6), fontsize=12, color=[0,0,0.2])
# %% [markdown]
airbnb_df = airbnb_df \
    .drop(columns=["weekly_price", "monthly_price"]) \
    .dropna()

airbnb_df
# %%
msno.bar(airbnb_df,figsize=(12, 6), fontsize=12, color='steelblue')
# %% [markdown]
# Cabe recalcar nuevamente que solo se incluirá información del suburbio de las
# viviendas por lo tanto los datos deben combinarse con la tabla
# `melb_suburb_df`. Esta es otra de las justificaciones por las cuales es
# conveniente normalizar relaciones de datos.
# %%
airbnb_df = airbnb_df.dropna().groupby("zipcode") \
    .agg(
        suburb_description_wordcount=("description", frequent_words),
        suburb_neighborhoods_overview_wordcount=("neighborhood_overview", frequent_words),
        suburb_transit_wordcount=("transit", frequent_words),
        suburb_rental_dailyprice=("price", "mean")
    ) \
    .reset_index() \
    .rename(columns={"zipcode": "suburb_postcode"})
# %%
airbnb_df
# %%
melb_suburb_df = melb_suburb_df.merge(
    airbnb_df, how='left', on="suburb_postcode"
)
# %%
msno.bar(melb_suburb_df,figsize=(12, 6), fontsize=12, color='steelblue')

# %%
