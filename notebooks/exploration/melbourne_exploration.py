# %% [markdown]
# # Diplomatura en Ciencas de Datos, Aprendizaje Automático y sus Aplicaciones
#
# Autores: Matias Oria, Antonela Sambuceti, Pamela Pairo, Benjamín Ocampo %%
# %% [markdown]
# Se trabajará sobre el conjunto de datos de [la compentencia
# Kaggle](https://www.kaggle.com/dansbecker/melbourne-housing-snapshot) para la
# estimación de precios de ventas de propiedades en Melbourne, Australia.
#
# Utilizaremos el conjunto de datos reducido producido por
# [DanB](https://www.kaggle.com/dansbecker siendo accedido por a través de un
# servidor de la Universidad Nacional de Córdoba para facilitar su acceso
# remoto.
# %% [markdown]
# ## Renombrado de columnas
# Debido a que se manipularán las columnas del dataframe a través de python se
# renombrarán los nombres asignados para que respeten los [estándares de
# código](https://www.python.org/dev/peps/pep-0008/) del lenguaje de
# programación y caractericen de mejor manera los datos almacenados.
# %%
import pandas as pd

URL = 'https://cs.famaf.unc.edu.ar/~mteruel/datasets/diplodatos/melb_data.csv'

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

def replace_columns(df, new_columns):
    new_col_names = {
        original_name: category + '_' + new_name
        for category, cols in new_columns.items()
        for original_name, new_name in cols.items()
    }
    return df.rename(columns=new_col_names)

melb_df = pd \
    .read_csv(URL) \
    .pipe(replace_columns, new_columns)

melb_df
# %% [markdown]
# ## Normalización del conjunto de datos
# Notar que se tiene incluido en un único *dataframe* información
# correspondiente del suburbio y las viviendas. Claramente los datos asociados a
# los suburbios se repiten por cada vivienda desaprovechando espacio de memoria
# y complejizando el estudio de la estructura del conjunto de datos. Por ende se
# puede [normalizar](https://en.wikipedia.org/wiki/Third_normal_form) la tabla
# `melb_df` separandola en dos `melb_suburb_df` y `melb_housing_df`,
# mantentiendo una *foreign key* en `melb_housing_df`. Esto permite unir ambas
# tablas por medio de la operación `join` y obtener el *dataframe* original. Si
# bien aún con esta organización no se obtiene una tercera forma normal, debido
# a que las regiones en `melb_suburb_df` también pueden ser tratadas de manera
# independiente, normalizar aún más complejizaría la realización de consultas
# sobre el conjunto de datos.
# %%
housing_cols = [col for col in melb_df if col.startswith("housing")]
suburb_cols = [col for col in melb_df if col.startswith("suburb")]

melb_suburb_df = melb_df[suburb_cols].drop_duplicates().reset_index(drop=True)
melb_housing_df = melb_df[housing_cols] \
    .join(melb_df["suburb_name"]) \
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
# %%
