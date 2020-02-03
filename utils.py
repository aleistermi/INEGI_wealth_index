
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

def request_CENSUS_data_from_INEGI(state):
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
    file.extractall()
    state_code_number = state_code[:2]
    dbf_name = "RESAGEBURB_{}DBF10.dbf".format(state_code_number)
    table = gpd.read_file(dbf_name)
    df = pd.DataFrame(table)
    os.remove('./{}'.format(dbf_name))
    df['CVEGEO'] = df['ENTIDAD'] + df['MUN'] + df['LOC'] + df['AGEB'] + df['MZA'] # Key for GeoJson file
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

def add_vars_wealth_index(df):

    '''
    Computes the variables that will be used for constructing the wealth index
    Inputs:
        df- DataFrame
    Returns:
        Dataframe
    '''

    as_percent = 100

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
