
import requests
import zipfile
import io
import geopandas as gpd
import pandas as pd
import os
from sklearn.impute import SimpleImputer
import numpy as np
from sklearn.decomposition import PCA
from sklearn import preprocessing
import json
from  impyute import fast_knn
import plotly
import plotly.express as px
from plotly.offline import download_plotlyjs, init_notebook_mode, iplot
def request_CENSUS_data_from_INEGI(state, path):
    '''
    It makes a request to download a zipfile containing a .dbf file, downloads and extracts the dbf, and
    creates a pandas dataframe

    Inputs:

        state (string): Name of the state to create the dataframe with. Valid values are:
                        CDMX- Mexico City
                        OAXACA- Oaxaca State
                        MORELOS- Morelos state
    Returns:

        dataframe of the selected state
    '''
    if state == "CDMX":
        state_code = '09_distrito_federal'
    if state == "OAXACA":
        state_code = '20_oaxaca'
    if state == "MORELOS":
        state_code = '17_morelos'

    data_URL = "https://www.inegi.org.mx/contenidos/programas/ccpv/2010/microdatos/iter/ageb_manzana/{}_2010_ageb_manzana_urbana_dbf.zip".format(state_code)
    req = requests.get(data_URL)
    file = zipfile.ZipFile(io.BytesIO(req.content))
    file.extractall(path=path)
    state_code_number = state_code[:2]
    dbf_name = "RESAGEBURB_{}DBF10.dbf".format(state_code_number)
    table = gpd.read_file(path + "/"+ dbf_name)
    df = pd.DataFrame(table)
    os.remove('./{}'.format( path + "/"+ dbf_name))
    df['CVEGEO'] = df['ENTIDAD'] + df['MUN'] + df['LOC'] + df['AGEB']# + df['MZA'] # Key for GeoJson file
    return df

def filter_by_geography(df, geographic_unit):
    '''
    filters by geographic unit and gets rid off rows that have zero population'''
    if geographic_unit == "manzana":
        df = df.copy()[(df['NOM_LOC']!='Total de la entidad') & (df['NOM_LOC']!='Total del municipio') & \
        (df['NOM_LOC']!='Total de la localidad urbana') & (df['NOM_LOC']!='Total AGEB urbana')]
        df['CVEGEO'] = df['ENTIDAD'] + df['MUN'] + df['LOC'] + df['AGEB']+ df['MZA']
    elif geographic_unit == "ageb":
        df = df.copy()[df['NOM_LOC'] == "Total AGEB urbana" ] # df_over_30_years = df.copy()[df['age']>30]
        df['CVEGEO'] = df['ENTIDAD'] + df['MUN'] + df['LOC'] + df['AGEB']
    elif geographic_unit == "localidad":
        df = df.copy()[df["NOM_LOC"] ==  'Total de la localidad urbana']
        df['CVEGEO'] = df['ENTIDAD'] + df['MUN'] + df['LOC']
    elif geographic_unit == "municipio":
        df = df.copy()[df['NOM_LOC'] == 'Total del municipio']
        df['CVEGEO'] = df['ENTIDAD'] + df['MUN']
    df.copy()[df['POBTOT'] == "0"]
    return df

def count_missing_manzanas(df):
    '''
    Counts the number of missing values (by manzana) for each of the selected variables

    Inputs:
            df- Dataframe with data at the manzana level
    Returns:
            Dataframe with percentage of missing values
    '''
    rv = {}
    for var in list(df.columns):
        d = {}
        d['missing'] = 0
        d['n/a'] = 0
        for element in list(df[var]):
            if element == '*':
                d['missing'] =  d['missing'] + 1
            elif element == 'N/D':
                d['n/a'] = d['n/a'] + 1
        d['missing'] = round( d['missing']/len(df) * 100,2)
        d['n/a'] =  round(d['n/a']/len(df) * 100,2)

        rv[var]= d
    return pd.DataFrame(rv).T

def convert_to_numeric(df, vars_to_numeric):
    '''
    Converts string values to numeric, making nulls those values with "N/D" or "*"
    Inputs:
            df: pandas dataframe
            vars_to_numeric: list of variable names
    Returns:
            Pandas dataframe
    '''
    for col in  vars_to_numeric:
        #df[col] = pd.to_numeric(df[col], errors='coerce')
        df.loc[:, col] = pd.to_numeric(df.loc[:, col], errors='coerce')
    return df


def impute_knn(df, numeric_vars, neighbors):
    X = convert_to_numeric(df, numeric_vars)
    X = df[numeric_vars].to_numpy()
    other_vars = list(set(df.columns) - set(numeric_vars) )
    X_strings = df[other_vars].reset_index()
    imputed_np = fast_knn(X, k= neighbors)

    X_imputed = pd.DataFrame.from_records(imputed_np, columns = numeric_vars)

    rv = X_strings.join(X_imputed)
    return rv

def add_vars_wealth_index(dataframe):

    '''
    Computes the variables that will be used for constructing the wealth index
    Inputs:
        df- DataFrame
    Returns:
        Dataframe
    '''
    numeric_vars = [ 'POBTOT','POBMAS','POBFEM', 'P15YM_AN', 'P15PRI_IN',
                    'P15YM_SE', 'POB15_64', 'POB65_MAS','VPH_DRENAJ', 'VPH_NODREN',
                    'VPH_S_ELEC', 'VPH_C_ELEC', 'VPH_AGUADV','VPH_AGUAFV', 'PRO_OCUP_C',
                    'VPH_PISOTI','VPH_PISODT', 'VPH_REFRI','VPH_AUTOM', 'VPH_INTER',
                    'VIVTOT', 'VPH_LAVAD', 'PHOG_IND']
    wealth_index_vars = ['illiterate', 'no_primary_educ', 'no_sewing', 'no_electricity' , 'no_water', 'dirt_floor',
                    'no_fridge', 'no_car', 'no_internet', 'overcrowded', 'indigenous_pop','no_washingmachine' ]
    df = convert_to_numeric(dataframe, numeric_vars)
    as_percent = 100
    try:
        df['illiterate'] = df['P15YM_AN']/(df['POB15_64'] + df['POB65_MAS']) * as_percent
        df['no_primary_educ'] = (df['P15PRI_IN'] + df['P15YM_SE']) /(df['POB15_64'] + df['POB65_MAS']) * as_percent
        df['no_sewing'] = (df['VPH_NODREN']) /(df['VPH_DRENAJ'] + df['VPH_NODREN']) * as_percent
        df['no_electricity'] = (df['VPH_S_ELEC']) /(df['VPH_S_ELEC'] + df['VPH_C_ELEC']) * as_percent
        df['no_water'] = (df['VPH_AGUAFV']) /(df['VPH_AGUADV'] + df['VPH_AGUAFV']) * as_percent
        df['dirt_floor'] = df['VPH_PISOTI']/(df['VPH_PISOTI'] + df['VPH_PISODT']) * as_percent
        df['no_fridge'] = (df['VIVTOT'] - df['VPH_REFRI']) / df['VIVTOT'] * as_percent
        df['no_car'] = (df['VIVTOT'] - df['VPH_AUTOM']) / df['VIVTOT'] * as_percent
        df['no_internet'] = (df['VIVTOT'] - df['VPH_INTER']) / df['VIVTOT'] * as_percent
        df['overcrowded'] = df['PRO_OCUP_C']
        df['no_washingmachine'] = (df['VIVTOT'] - df['VPH_LAVAD']) / df['VIVTOT'] * as_percent
        df ['indigenous_pop'] = (df['PHOG_IND']) / df['POBTOT'] * as_percent
    except:
        pass
    for var in wealth_index_vars:
        for i,r in df.iterrows():
            mean = df[var].mean()
            if r[var]<0 or r[var]>100:
                df.at[i, var] = mean

    return df

def build_query_for_nulls(list_of_vars):
    '''
    Creates a query to filter a data frame with multiple variables not containing nulls
    '''
    string = ""
    for var in list_of_vars[:-1]:
        var = var + '.notnull() and '
        string += var
    string = string + list_of_vars[-1] + '.notnull()'
    return string

def construct_index(df, wealth_index_vars):

    df_no_missings = df.query(build_query_for_nulls(wealth_index_vars), engine = 'python')
    X = df_no_missings[wealth_index_vars].to_numpy()
    normalized_X = preprocessing.normalize(X)
    pca = PCA(n_components = 1)
    pca = pca.fit_transform(X)
    principalDf = pd.DataFrame(data = pca, columns = ['pca_1', ])
    rv = df_no_missings.merge(principalDf, left_index=True, right_index = True)

    min_pca1 =  pca.min() # Index 0-1
    max_pca1 = pca.max()
    difference = max_pca1 - min_pca1
    rv['index_01'] = (rv['pca_1'] - min_pca1)/difference
    return rv

def impute_values(df, imp_strategy, neighbors, numeric_vars):

    X = convert_to_numeric(df, numeric_vars)
    X = df[numeric_vars].to_numpy()
    other_vars = list(set(df.columns) - set(numeric_vars) )
    X_strings = df[other_vars].reset_index(drop=True)
    if imp_strategy == "knn":
        # imputer = KNNImputer(n_neighbors = neighbors, weights = weight_type)
        # imputed = imputer.fit_transform(X) # This is very costly
# from here https://impyute.readthedocs.io/en/master/api/cross_sectional_imputation.html
# https://impyute.readthedocs.io/en/master/api/cross_sectional_imputation.html
        imputed = fast_knn(X, k= neighbors)
    else:
        imputer = SimpleImputer(missing_values = np.nan, strategy = imp_strategy)
        imputer.fit(X)
        imputed = imputer.transform(X)
    X_imputed = pd.DataFrame.from_records(imputed, columns = numeric_vars)
    rv = X_strings.join(X_imputed)
    return rv

def convert_transform_shp (path, name_geojson):

    '''
    09a- AGEB
    09ar - Rural Agebs
    09cd - localidad rural
    09e - streets (lines)
    '''
    data = gpd.read_file(path, encoding='windows-1252' )
    data['geometry'] = data['geometry'].to_crs(epsg=4326)
    data.to_file(name_geojson, driver="GeoJSON")
    with open(name_geojson, 'r') as f:
        json_file = json.load(f)
    return json_file, data

def create_choropleth (df, geojson_file, variable, color_scale, color_range,
                      label, opacity_scale, zoom, pathsave, street_map, geography, opacity = 0.4 ):

    fig = px.choropleth_mapbox(df, geojson=geojson_file, locations='CVEGEO', color = variable,
                           color_continuous_scale= color_scale,
                           range_color = (color_range[0], color_range[1]),
                           featureidkey="properties.CVEGEO",
                           mapbox_style="carto-positron",
                           zoom=zoom, center = {"lat": 19.36, "lon": -99.133209},
                           opacity= opacity_scale,
                           labels={variable:label}
                          )
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    fig.update_layout(
        title_text='Spatial Distribution of Marginalization Index',

    )
    if geography == True:
        fig.update_geos(
            scope="south america",
            showcountries=True, countrycolor="Black",
            showsubunits=True, subunitcolor="Black"
        )
    if street_map == True:
        fig.update_layout(mapbox_style="open-street-map")

    fig.write_html(pathsave)
    fig.show()


########################################################
# To review #
########################################################
# key mapbox pk.eyJ1IjoiYWxlaXN0ZXJtIiwiYSI6ImNrNjVjMnFiNjA2eGgzbXF3a2V1dmN3eGUifQ.zNzPm9NqQKFDksXdgU-eeA
# vars_to_impute = []
# for i, r in (missing_val_CDMX[missing_val_CDMX['missing']>0]).iterrows():
#     vars_to_impute.append(i)
# # Keep manzanas with no missing values
#df_no_missings = knn_imputed.query(utils.build_query_for_nulls(wealth_index_vars), engine = 'python')
# def impute_values_1 (df, imp_strategy, numeric_vars):
#
#     X = convert_to_numeric(df, numeric_vars)
#     X = df[numeric_vars].to_numpy()
#     other_vars = list(set(df.columns) - set(numeric_vars) )
#     X_strings = df[other_vars].reset_index()
#
#     imputer = SimpleImputer(missing_values = np.nan, strategy = imp_strategy)
#     imputer.fit(X)
#     imputed = imputer.transform(X)
#     X_imputed = pd.DataFrame.from_records(imputed, columns = numeric_vars)
#     rv = X_strings.join(X_imputed)
#     return rv
#
# def create_geojson_from_shp (path, name):
#
#     reader = shapefile.Reader(path, encoding='windows-1252')
#     fields = reader.fields[1:]
#     field_names = [field[0] for field in fields]
#     buffer = []
#     for sr in reader.shapeRecords():
#         atr = dict(zip(field_names, sr.record))
#         geom = sr.shape.__geo_interface__
#         buffer.append(dict(type="Feature", geometry=geom, properties=atr))
#
#     # # write the GeoJSON file
#
#     geojson = open(name, "w")
#     geojson.write(dumps({"type": "FeatureCollection", "features": buffer}, indent=2) + "\n")
#     geojson.close()
#     with open(name, 'r') as f:
#         geojsonfile = json.load(f)
#     return geojsonfile
#
# fig = px.choropleth_mapbox(df_with_index, geojson=geojson_file, locations='CVEGEO', color='index_01',
#                            color_continuous_scale="bluered",
#                            range_color=(0, 1),
#                            featureidkey="properties.CVEGEO",
#                            mapbox_style="carto-positron",
#                            zoom=9.5, center = {"lat": 19.36, "lon": -99.133209},
#                            opacity=0.75,
#                            labels={'index_01':'wealth index (0= Richer, 1 =  Poorer)'}
#                           )
# fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
# # fig.update_layout(
# #     title_text='Spatial Distribution of Marginalization Index',
#
# # )
#
# # fig.update_geos(
# #     visible=False, resolution=50, scope="north america",
# #     showcountries=True, countrycolor="Black",
# #     showsubunits=True, subunitcolor="Blue"
# # )
# fig.update_layout(mapbox_style="open-street-map")
#
# # fig.update_layout(
# #     mapbox = {
# #         'style': "stamen-terrain",
# #         'center': {"lat": 19.432608, "lon": -99.133209},
# #         'zoom': 10},
# #     showlegend = False)
#
# fig.show()
# fig.write_html("./one_map.html")
# def convert_transform_shp (path, name_geojson):
#
#     data = gpd.read_file(path, encoding='windows-1252' )
#     data['geometry'] = data['geometry'].to_crs(epsg=4326)
#     data.to_file(name_geojson, driver="GeoJSON")
#     with open(name_geojson, 'r') as f:
#         json_file = json.load(f)
#     return json_file
# def read_shapefile(sf):
#     """
#     Read a shapefile into a Pandas dataframe with a 'coords'
#     column holding the geometry information. This uses the pyshp
#     package
#     """
#     fields = [x[0] for x in sf.fields][1:]
#     records = sf.records()
#     shps = [s.points for s in sf.shapes()]
#     df = pd.DataFrame(columns=fields, data=records)
#     df = df.assign(coords=shps)
#     return df
