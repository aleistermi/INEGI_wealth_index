
import requests
import zipfile
import io
import geopandas as gpd
import pandas as pd
import os
#from sklearn.impute import SimpleImputer
import folium
import numpy as np
from sklearn.decomposition import PCA
from sklearn import preprocessing
import json
from  impyute import fast_knn
import plotly
import plotly.express as px
import matplotlib.pyplot as plt
from plotly.offline import download_plotlyjs, init_notebook_mode, iplot
from sklearn.impute import KNNImputer

# Variables we want to keep to construct the wealth index
IDENTIFIER_vars = ['entidad','nom_ent','mun','nom_mun','loc','nom_loc',
    'ageb','mza','cvegeo']

POPULATION_vars= ['pobtot','pobmas','pobfem', 'pob0_14', 'pob15_64',
'pob65_mas', 'vivtot']

ASSET_OWNERSHIP_vars = [ 'vph_radio', 'vph_tv', 'vph_refri','vph_autom',
'vph_inter', 'vph_lavad', 'vph_pc', 'vph_telef', 'vph_cel']

asset_ownsership_vars = ['no_radio', 'no_tv','no_fridge', 'no_car',
'no_internet','no_washingmachine', 'no_pc', 'no_telephone', 'no_cellphone']

all_variables_index = asset_ownsership_vars

numeric_vars = POPULATION_vars +  ASSET_OWNERSHIP_vars
vars_to_keep = IDENTIFIER_vars + POPULATION_vars + ASSET_OWNERSHIP_vars

states = {'01': 'Aguascalientes','02': 'Baja California','03': 'Baja California Sur','04':
 'Campeche','05': 'Coahuila','06': 'Colima','07': 'Chiapas','08': 'Chihuahua','09': 'Cdmx',
 '10': 'Durango','11': 'Guanajuato','12': 'Guerrero','13': 'Hidalgo','14': 'Jalisco',
 '15': 'México','16': 'Michoacán','17': 'Morelos','18': 'Nayarit','19': 'Nuevo León','20': 'Oaxaca',
 '21': 'Puebla','22': 'Querétaro','23': 'Quintana Roo','24': 'San Luis Potosí','25': 'Sinaloa',
 '26': 'Sonora','27': 'Tabasco','28': 'Tamaulipas','29': 'Tlaxcala','30': 'Veracruz',
 '31': 'Yucatán','32': 'Zacatecas'}

def request_CENSUS_data_from_INEGI(state, path, save_local =  False):
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
    data_URL = "https://www.inegi.org.mx/contenidos/programas/ccpv/2010/datosabiertos/ageb_y_manzana/resageburb_{}_2010_csv.zip".format(state)

    req = requests.get(data_URL)
    file = zipfile.ZipFile(io.BytesIO(req.content))
    file.extractall(path = path)
    file_name = "resultados_ageb_urbana_{}_cpv2010.csv".format(state)
    table = pd.read_csv(path + "/"+ 'resultados_ageb_urbana_{}_cpv2010/'.format(state)+ 'conjunto_de_datos/' + file_name)
    table['mun'] =  table['mun'].apply(lambda x: "{:03d}".format(x))
    table['loc'] = table['loc'].apply(lambda x: "{:04d}".format(x))
    table['mza'] = table['mza'].apply(lambda x: "{:03d}".format(x))
    table['entidad'] = table['entidad'].apply(lambda x: "{:02d}".format(x))
    for i, r in table.iterrows():
        if type(r['ageb']) == int:
            table.at[i,'ageb'] = str(r['ageb']).zfill(4)
        # if type(r['loc']) == int:
        #     table.at[i,'loc'] = str(r['loc']).zfill(4)
        # table.at[i,'mun'] = str(r['mun']).zfill(3)
        # table.at[i,'entidad'] = str(r['ageb']).zfill(2)
        # table.at[i,'mza'] = str(r['ageb']).zfill(3)
        # table.at[i,'loc'] = str(r['loc']).zfill(4)
    if save_local == False:
        os.remove('./{}'.format( path + "/"+ 'resultados_ageb_urbana_{}_cpv2010/'.format(state)+ 'conjunto_de_datos/' + file_name))
    return table

def filter_by_geography(df, geographic_unit):
    '''
    filters by geographic unit and gets rid off rows that have zero population
    inputs:
        df - Dataframe from census
        geographic_unit(string) - Geographic unit to keep '''
    if geographic_unit == "manzana":
        df = df.copy()[(df['nom_loc']!='Total de la entidad') & \
        (df['nom_loc']!='Total del municipio') & \
        (df['nom_loc']!='Total de la localidad urbana') & \
        (df['nom_loc']!='Total AGEB urbana')]

        df['cvegeo'] = df['entidad'] + df['mun'] + df['loc'] + \
                       df['ageb']+ df['mza']
        df['block'] = df['entidad'] + df['mun'] + df['loc'] + \
                      df['ageb'] + df['mza']

    elif geographic_unit == "ageb":
        df = df.copy()[(df['nom_loc']=='Total AGEB urbana')]  # df_over_30_years = df.copy()[df['age']>30]
        df['cvegeo'] = df['entidad'] + df['mun'] + df['loc'] + df['ageb']
        df['ageb'] = df['entidad'] + df['mun'] + df['loc'] + df['ageb']

    elif geographic_unit == "localidad":
        df = df.copy()[df["nom_loc"] ==  'Total de la localidad urbana']
        df['cvegeo'] = df['entidad'] + df['mun'] + df['loc']
    elif geographic_unit == "municipio":
        df = df.copy()[df['NOM_loc'] == 'Total del municipio']
        df['cvegeo'] = df['entidad'] + df['mun']
    return df

def count_missings(df):
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
        d['str_missing'] = 0

        for element in list(df[var]):
            if element == '*':
                d['str_missing'] =  d['str_missing'] + 1
        d['str_missing'] = round( d['str_missing']/len(df) * 100,2)

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
    # for col in vars_to_numeric:
    #     #df[col] = pd.to_numeric(df[col], errors='coerce')
    #     df.loc[:, col] = pd.to_numeric(df.loc[:, col], errors='coerce')
    df[vars_to_numeric] = df[vars_to_numeric].apply(pd.to_numeric, errors='coerce', axis = 1)

    return df


def impute_knn(df, numeric_vars, neighbors):
    X = convert_to_numeric(df, numeric_vars)
    X = df[numeric_vars]
    X = X.values
    other_vars = list(set(df.columns) - set(numeric_vars) )
    X_strings = df[other_vars] #.reset_index()
    imputed_np = fast_knn(X, k = neighbors)

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
    AS_PERCENT = 100

    # asset_ownsership_vars
    df['no_fridge'] = (df['vivtot'] - df['vph_refri']) / df['vivtot'] * AS_PERCENT #5
    df['no_car'] = (df['vivtot'] - df['vph_autom']) / df['vivtot'] * AS_PERCENT #6
    df['no_internet'] = (df['vivtot'] - df['vph_inter']) / df['vivtot'] * AS_PERCENT #7
    df['no_washingmachine'] = (df['vivtot'] - df['vph_lavad']) / df['vivtot'] * AS_PERCENT #8
    df['no_radio'] = (df['vivtot'] - df['vph_radio']) / df['vivtot'] * AS_PERCENT #9
    df['no_tv'] = (df['vivtot'] - df['vph_tv']) / df['vivtot'] * AS_PERCENT #10
    df['no_pc'] = (df['vivtot'] - df['vph_pc']) / df['vivtot'] * AS_PERCENT #11
    df['no_cellphone'] = (df['vivtot'] - df['vph_cel']) / df['vivtot'] * AS_PERCENT #11
    df['no_telephone'] = (df['vivtot'] - df['vph_telef']) / df['vivtot'] * AS_PERCENT #11


    # household_characteristic

    return df

def impute_values(df, imp_strategy, neighbors, numeric_vars):

    X = convert_to_numeric(df, numeric_vars)
    X = df[numeric_vars].to_numpy()
    other_vars = list(set(df.columns) - set(numeric_vars) )
    X_strings = df[other_vars].reset_index(drop=True)
    if imp_strategy == "knn":
        imputer = KNNImputer(n_neighbors = neighbors) #weights = weight_type
        imputed = imputer.fit_transform(X) # This is very costly
# from here https://impyute.readthedocs.io/en/master/api/cross_sectional_imputation.html
# https://impyute.readthedocs.io/en/master/api/cross_sectional_imputation.html
#         imputed = fast_knn(X, k= neighbors)
    else:
        imputer = SimpleImputer(missing_values = np.nan, strategy = imp_strategy)
        imputer.fit(X)
        imputed = imputer.transform(X)
    X_imputed = pd.DataFrame.from_records(imputed, columns = numeric_vars)
    rv = X_strings.join(X_imputed)
    return rv

###########################################
# NO NEED OF THIS #
###########################################
def numeric_missings(df):
    den = len(df)
    df = pd.DataFrame(df.isnull().sum(), columns = ['# missings'])
    df['# missings'] = round(df['# missings']/den*100,2)
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

def construct_index(df, wealth_index_vars,  impute_method):
    '''
    It adds to a dataframe the columns corresponding to the wealth index.
    For each list of variables in the argument "wealth_index_vars", it creates
    two new columns: one for the value of the first principal component, and one
    for the re-scaled 0-1 value of the first component. Each column is indexed
    with numbers from 0 to len(wealth_index_vars).

    Inputs:
        df(dataframe) : dataframe with
    '''
    ##
    IDENTIFIER_vars = ['entidad','nom_ent','mun','nom_mun','loc','nom_loc',
    'ageb','mza','cvegeo', 'block']
    POPULATION_vars= ['pobtot','pobmas','pobfem', 'pob0_14', 'pob15_64',
    'pob65_mas', 'vivtot']

    ASSET_OWNERSHIP_vars = [ 'vph_radio', 'vph_tv', 'vph_refri','vph_autom',
    'vph_inter', 'vph_lavad', 'vph_pc', 'vph_telef', 'vph_cel']

    asset_ownsership_vars = ['no_radio', 'no_tv','no_fridge', 'no_car',
    'no_internet','no_washingmachine', 'no_pc', 'no_telephone', 'no_cellphone']

    all_variables_index = asset_ownsership_vars

    numeric_vars = POPULATION_vars +  ASSET_OWNERSHIP_vars
    vars_to_keep = IDENTIFIER_vars + POPULATION_vars + ASSET_OWNERSHIP_vars + asset_ownsership_vars

    #Keep relevant variables only
    df = df[vars_to_keep].reset_index(drop=True)
    # These are missing rows
    df = filter_missing_rows(df)
    # Convert them to numeric
    df = convert_to_numeric(df, numeric_vars)
    # Add wealth index vars
    df = add_vars_wealth_index(df)

    # Impute data to the local median
    if impute_method == "local_median":
        df = df.groupby("ageb").apply(lambda x: x.fillna(x.median())).\
        reset_index(drop=True)
    elif impute_method == "global_median":
        for ls in wealth_index_vars:
            for var in ls:
                df[var].fillna((df[var].median()), inplace=True)
    elif impute_method == "knn":
        print("imputing knn")
        df = impute_values(df, "knn", 4, numeric_vars )

    # Estimate index
    i = 0
    list_of_dfs = []
    for ls in wealth_index_vars:
        df = df.query(build_query_for_nulls(ls), engine = 'python')
        X = df[ls].values
        normalized_X = preprocessing.normalize(X)
        pca = PCA(n_components = 1)
        pca = pca.fit_transform(X)
        first_component = pd.DataFrame(data = pca, columns = \
        ['pca_{}'.format(i), ])
        min_pca =  pca.min() # Index 0-1
        max_pca = pca.max()
        difference = max_pca - min_pca
        first_component['index_{}'.format(i)] = \
        (first_component['pca_{}'.format(i)] - min_pca)/difference
        list_of_dfs.append(first_component)
        i = i+1

    for indices in list_of_dfs:
        df = df.merge(indices, left_index=True, right_index = True)

    return df

def plot_simple_histogram(series, name, bins):
    '''
    Plots simple histograms
    Inputs:
        series (series) - Column of pandas dataframe (or any other series)
        name (str) -  Name of the plot (title)
        bins (int) -  Number of bins of the histogram
    '''
    n, bins, patches = plt.hist(series, bins=bins)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    #plt.xlim(-10, 2000)
    plt.title('Distribution of {}'.format(name))

    # Tweak spacing to prevent clipping of ylabel
    plt.savefig('./plots/{}.png'.format(name))
    plt.show()


def convert_transform_shp (path, name_geojson):

    '''
    09a- ageb
    09ar - Rural agebs
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
                      label, opacity_scale, zoom, pathsave, street_map, geography, show = False ):

    fig = px.choropleth_mapbox(df, geojson=geojson_file, locations='block', color = variable,
                           color_continuous_scale= color_scale,
                           range_color = (color_range[0], color_range[1]),
                           featureidkey="properties.block",
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
    if show == True:
        fig.show()

def folium_manzanas_choropleth(geojsonfile, dataframe, index_to_plot, color, legend_title, filename, fill_opacity_ =
                            0.7, line_opacity_ = 0.2, save = True):

    '''
    colors: 'BuGn', 'BuPu', 'GnBu', 'OrRd', 'PuBu', 'PuBuGn', 'PuRd', 'RdPu', 'YlGn', 'YlGnBu', 'YlOrBr',
    'YlOrRd', 'BrBg', 'PiYG', 'PRGn', 'PuOr', 'RdBu', 'RdGy', 'RdYlBu', 'RdYlGn', 'Spectral', 'Accent', 'Dark2'
    'Paired', 'Pastel1', 'Pastel2', 'Set1', 'Set2', 'Set3'

    '''

    m = folium.Map(location=[19.36, -99.133209], zoom_start=11.5, fill_opacity_ = 0.7, line_opacity_ = 0.2 )

    # Add the color for the chloropleth:
    m.choropleth(
     geo_data = geojsonfile,
     name = 'choropleth',
     data = dataframe,
     columns=['block', index_to_plot],
     key_on = 'properties.block',
     fill_color = color,
     fill_opacity = fill_opacity_,
     line_opacity = line_opacity_,
     legend_name=legend_title)
    folium.LayerControl().add_to(m)
    m.save('plots/{}.html'.format(filename))
    m.save("plots/{}.png".format(filename))

def plot_missings(df, title):

    '''
    Plots missing values denoted with a "*"
    inputs: df
    returns: matplotlib object
    '''
    missings = count_missings(df[list(set(vars_to_keep) - set(IDENTIFIER_vars))])
    missings['variable'] = missings.index
    missings = missings.sort_values(by=['str_missing'], ascending=True).reset_index(drop=True)
    plt.barh(missings.index, missings['str_missing'], align='center', alpha=0.5)
    plt.yticks(missings.index, missings['variable'], size=10)
    plt.xlabel('(% from total)')
    plt.title(title)
    #plt.tick_params(direction='in', size = 2, length=6, width=2,  grid_alpha=0.5)
    plt.tight_layout()
    plt.savefig("./plots/barchart_missings.png", dpi=300)
    plt.show()
    return plt

def horizontal_barplot(df, variable, title, ylabel, color, horizontal = True):

    categories = list(df[ylabel])
    y_pos = np.arange(len(categories))
    count = df[variable]
    if horizontal:
        plt.barh(y_pos, count, align='center',
                 alpha=0.5, color = color)
        plt.yticks(y_pos, categories)
        plt.ylabel(ylabel)
    else:
        plt.bar(y_pos, count, align='center',
                 alpha=0.5, color = color)
        plt.xticks(y_pos, categories)

        plt.xlabel(ylabel)
    plt.title(title)

    plt.show()


def convert_transform_shp (path, name_geojson):

    data = gpd.read_file(path, encoding='windows-1252' )
    data['geometry'] = data['geometry'].to_crs(epsg=4326)
    data.to_file(name_geojson, driver="GeoJSON")
    with open(name_geojson, 'r') as f:
        json_file = json.load(f)
    return json_file, data


def filter_missing_rows(df):

    df = df[~((df['pobmas']=="*") & (df['pobfem']=="*") &
     (df['pob15_64']=="*")  &
       (df['pob65_mas']=="*") & (df['pob0_14']=="*") &
       (df['vph_radio']=="*") &(df['vph_tv']=="*") &
       (df['vph_refri']=="*") &
       (df['vph_autom']=="*")  & (df['vph_inter']=="*") &
       (df['vph_lavad']=="*") & (df['vph_pc']=="*") &
       (df['vph_telef']=="*")& (df['vph_cel']=="*"))].reset_index(drop=True)
    return df

def filter_geojson_by_state(geojson, state_code):
    '''
    Filters a geojson file for all the country, by statecode
    CDMX code is "09"
    '''
    d = {}
    d['type'] = 'FeatureCollection'
    d['crs'] = {'type': 'name', 'properties': {'name': 'urn:ogc:def:crs:OGC:1.3:CRS84'}}
    d['features'] = []
    for element in geojson['features']:
        for k,v in element.items():
            if k == "properties":
                if v["munIC"][:2] == state_code:
                    d['features'].append(element)
    return d
