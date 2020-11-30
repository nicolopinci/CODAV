# -*- coding: utf-8 -*-

# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

import base64
import datetime
import io

import dash
from dash.dependencies import Input, Output, State, MATCH, ALL

import dash_extendable_graph as deg
import dash_core_components as dcc
import dash_html_components as html
import plotly
import plotly.express as px
import pandas as pd
import dash_table
import dash_daq as daq
import dash_draggable
import json
import pickle
import time
import numpy as np
import os
from math import log10, floor, ceil, sqrt
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.tsa.vector_ar.var_model import VAR
import scipy
from plotly.subplots import make_subplots
import heapq


import sklearn as sk
from sklearn.linear_model import LinearRegression

import plotly.graph_objs as go
from dash.exceptions import PreventUpdate

# Import libraries for predictions
import pmdarima as pm
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.model_selection import TimeSeriesSplit
from statsmodels.tsa.statespace.sarimax import SARIMAX
from fbprophet import Prophet

# Central map
import dash_leaflet as dl
import dash_leaflet.express as dlx



import urllib.request


covid_data = None
dimensions = []
graph_infos = []
colorblind_colors = ['rgb(27,158,119)','rgb(217,95,2)','rgb(117,112,179)']
topmap_colors = ['#ffffb2','#fecc5c','#fd8d3c','#e31a1c']



external_stylesheets = ["static/style.css", "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.14.0/css/all.min.css"]
external_scripts = ["static/moveGraphs.js", "https://ajax.googleapis.com/ajax/libs/jquery/2.1.1/jquery.min.js", "static/loader.js"]

app = dash.Dash(__name__, external_stylesheets=external_stylesheets, external_scripts=external_scripts)

server = app.server


class GraphInfo:
    def __init__(self, dataset, title, prediction_method = "", std = 2, go_type = "Scatter", divide_traces =False, additional_columns = 'exog_var[["date", "stringency_index"]]', graph_type = "line", same_color = False, axes = [], color = None, filters = [], animation = None, map_type = 'choropleth', location_mode = 'country names', colorscale= [(0, "#ffeda0"), (0.5, "#feb24c"), (1, "#f03b20")], animation_frame = 'data["date"].astype(str)', min_animation = None, max_animation = None, plot_type = "lines", hide_side_legend = False, width = 650, height = 500):
        self.title = title 
        self.axes = axes
        self.color = color
        self.filters = filters
        self.animation = animation
        self.graph_type = graph_type
        self.dataset = dataset
        self.map_type = map_type
        self.location_mode = location_mode
        self.colorscale = colorscale
        self.animation_frame = animation_frame
        self.min_animation = min_animation
        self.max_animation = max_animation
        self.plot_type = plot_type
        self.hide_side_legend = hide_side_legend
        self.width = width
        self.height = height
        self.go_type = go_type
        self.same_color = same_color
        self.additional_columns = additional_columns
        self.prediction_method = prediction_method
        self.divide_traces = divide_traces
        self.std = std

    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True, indent=4)

class Axis:
    def __init__(self, label = "", content = [], labels = [], log_scale = False):
        self.label = label
        self.log_scale = log_scale
        self.content = content
        self.labels = labels

class Filter:
    def __init__(self, column_name = "", default_value = [], multi = True, prec = 0, filter_type="Dropdown", start_date = "2020-01-01", end_date="2020-12-31", show_on_marker = False, filter_name = "", n_steps = None):
        self.column_name = column_name
        self.default_value = default_value
        self.multi = multi
        self.filter_type = filter_type
        self.start_date = start_date
        self.end_date = end_date
        self.show_on_marker = show_on_marker
        self.filter_name = filter_name
        self.n_steps = n_steps
        self.prec = prec

class Animation:
    def __init__(self, active = False, axis_number = None):
        self.active = active
        self.axis_number = axis_number



# Serving local static files
@app.server.route('/static/<path:path>')
def static_file(path):
    static_folder = os.path.join(os.getcwd(), 'static')
    return send_from_directory(static_folder, path)


# see https://plotly.com/python/px-arguments/ for more options

project_name = "CODAV"

def mask_max(in_data, filter_key):

    in_data[filter_key].fillna(in_data[filter_key].min())
    in_data[filter_key].fillna(0)

    return in_data[in_data[filter_key] == in_data[filter_key].max()]
   

def generate_layout():
    return html.Div(
    id="top_container",
    children=[

        html.Div(id = "top_bar", children = [
	    dcc.Upload(
        id='upload-data',
        children=html.Div([
            html.I("", className="fas fa-upload")
        ]),
               
        # Do not allow multiple files to be uploaded
        multiple=False
        ),


        dcc.Upload(
        id='upload-school',
        children=html.Div([
              html.I("", className="fas fa-user-graduate")
        ]),

        # Do not allow multiple files to be uploaded
        multiple=False
        ),

      


        html.Div(id='output-data-upload'),
        html.H1(children=[html.I("", className="fas fa-virus"), html.Span("   "), html.Span(project_name)])
        ]),
        
        
        html.Div(id = "leftSide", children = []),
        html.Div(id = "centralMap", children = []),
        html.Div(id = "rightSide", children = []),

         daq.BooleanSwitch(
          id = "color_blind",
          on=False,
          label="Color-blind",
          labelPosition="botttom"
          ),

        html.Ul(id = "three_buttons", children = [html.Li(id = "general_button", className="currentlySelected", children = ["General analyses"]), html.Li(id = "edu_button", children = ["Education"]), html.Li(id = "pred_button", children = ["Predictions"])]),
        html.Div(id="analyses_container", className = "graphs_container", children=[]),
        html.Div(id="edu_container", className = "graphs_container", children=[]),
        html.Div(id="predictions_container", className = "graphs_container", children=[]),

   

        html.Div(id="filter_equivalence", style={'display': 'none'}),

        # Hidden div inside the app that stores the intermediate value
        html.Div(id='saved_data', style={'display': 'none'}),
        html.Div(id='saved_school', style={'display': 'none'})

    ],
    )

app.layout = generate_layout

# Display different tabs

@app.callback(Output('predictions_container', 'style'), [Input('pred_button', 'n_clicks')], prevent_initial_call = True)
def change_button_style(n_clicks):

    if n_clicks % 2 == 1:
        return {'display': 'block'}
    else:
        return {'display': 'none'}



@app.callback(Output('edu_container', 'style'), [Input('edu_button', 'n_clicks')], prevent_initial_call = True)
def change_button_style(n_clicks):

    if n_clicks % 2 == 1:
        return {'display': 'block'}
    else:
        return {'display': 'none'}


@app.callback(Output('analyses_container', 'style'), [Input('general_button', 'n_clicks')], prevent_initial_call = True)
def change_button_style(n_clicks):

    if (n_clicks + 1) % 2 == 1:
        return {'display': 'block'}
    else:
        return {'display': 'none'}



# Colour the buttons

@app.callback(Output('pred_button', 'className'), [Input('pred_button', 'n_clicks')], prevent_initial_call = True)
def change_button_style(n_clicks):

    if n_clicks % 2 == 1:
        return "currentlySelected"
    else:
        return "";


@app.callback(Output('edu_button', 'className'), [Input('edu_button', 'n_clicks')], prevent_initial_call = True)
def change_button_style(n_clicks):

    if n_clicks % 2 == 1:
        return "currentlySelected"
    else:
        return "";


@app.callback(Output('general_button', 'className'), [Input('general_button', 'n_clicks')], prevent_initial_call = True)
def change_button_style(n_clicks):

    if (n_clicks + 1) % 2 == 1:
        return "currentlySelected"
    else:
        return "";



def parse_contents(covid, school):
    covid_content_type, covid_content_string = covid.split(',')
    school_content_type, school_content_string = school.split(',')

    covid_decoded = base64.b64decode(covid_content_string)
    school_decoded = base64.b64decode(school_content_string)

    school = pd.read_csv(io.StringIO(school_decoded.decode('utf-8')))
    covid = pd.read_csv(io.StringIO(covid_decoded.decode('utf-8')))

    school['date'] = pd.to_datetime(school['date'], format = "%d/%m/%Y")
    covid['date'] = pd.to_datetime(covid['date'])

    school['date'] = school['date'].dt.strftime("%Y-%m-%d")
    covid['date'] = covid['date'].dt.strftime("%Y-%m-%d")

    school["Physical_education"] = pd.Series(dtype = 'float')
    school.loc[school["Status"] == "Fully open", "Physical_education"] = 1
    school.loc[school["Status"] == "Partially open", "Physical_education"] = 1.5
    school.loc[school["Status"] == "Closed due to COVID-19", "Physical_education"] = 2


    # Pre-processing
    school = school.dropna(axis=0, subset=['Status'])
    covid["total_cases"] = covid.groupby("location")["total_cases"].apply(lambda x: x.ffill().fillna(0))
    covid["new_cases"] = covid.groupby("location")['new_cases'].apply(lambda x: x.fillna(0))

    covid["new_cases_smoothed"] = covid.groupby("location")["new_cases_smoothed"].apply(lambda x: x.fillna(0))
    covid["total_deaths"] = covid.groupby("location")["total_deaths"].apply(lambda x: x.ffill().fillna(0))
    covid["new_deaths"] = covid.groupby("location")["new_deaths"].apply(lambda x: x.fillna(0))
    covid["new_deaths_smoothed"] = covid.groupby("location")["new_deaths_smoothed"].apply(lambda x: x.fillna(0))
    covid["total_cases_per_million"] = covid.groupby("location")["total_cases_per_million"].apply(lambda x: x.ffill().fillna(0))
    covid["new_cases_per_million"] = covid.groupby("location")["new_cases_per_million"].apply(lambda x: x.fillna(0))
    covid["new_cases_smoothed_per_million"] = covid.groupby("location")["new_cases_smoothed_per_million"].apply(lambda x: x.fillna(0))
    covid["total_deaths_per_million"] = covid.groupby("location")["total_deaths_per_million"].apply(lambda x: x.ffill().fillna(0))
    covid["new_deaths_per_million"] = covid.groupby("location")["new_deaths_per_million"].apply(lambda x: x.fillna(0))
    covid["new_deaths_smoothed_per_million"] = covid.groupby("location")["new_deaths_smoothed_per_million"].apply(lambda x: x.fillna(0))
    covid["reproduction_rate"] = covid.groupby("location")["reproduction_rate"].bfill().fillna(0)
    covid["icu_patients"] = covid.groupby("location")["icu_patients"].apply(lambda x: x.fillna(0))
    covid["icu_patients_per_million"] = covid.groupby("location")["icu_patients_per_million"].apply(lambda x: x.fillna(0))
    covid["hosp_patients"] = covid.groupby("location")["hosp_patients"].apply(lambda x: x.fillna(0))
    covid["hosp_patients_per_million"] = covid.groupby("location")["hosp_patients_per_million"].apply(lambda x: x.fillna(0))
    covid["weekly_icu_admissions"] = covid.groupby("location")["weekly_icu_admissions"].apply(lambda x: x.fillna(0))
    covid["weekly_icu_admissions_per_million"] = covid.groupby("location")["weekly_icu_admissions_per_million"].apply(lambda x: x.fillna(0))
    covid["weekly_hosp_admissions"] = covid.groupby("location")["weekly_hosp_admissions"].apply(lambda x: x.fillna(0))
    covid["weekly_hosp_admissions_per_million"] = covid.groupby("location")["weekly_hosp_admissions_per_million"].apply(lambda x: x.fillna(0))
    covid["total_tests"] = covid.groupby("location")["total_tests"].apply(lambda x: x.ffill().fillna(0))
    covid["total_tests_per_thousand"] = covid.groupby("location")["total_tests_per_thousand"].apply(lambda x: x.ffill().fillna(0))
    covid["new_tests"] = covid.groupby("location")["new_tests"].apply(lambda x: x.fillna(0))
    covid["new_tests_per_thousand"] = covid.groupby("location")["new_tests_per_thousand"].apply(lambda x: x.fillna(0))
    covid["new_tests_smoothed"] = covid.groupby("location")["new_tests_smoothed"].apply(lambda x: x.fillna(0))
    covid["new_tests_smoothed_per_thousand"] = covid.groupby("location")["new_tests_smoothed_per_thousand"].apply(lambda x: x.fillna(0))
    covid["tests_per_case"] = covid.groupby("location")["tests_per_case"].apply(lambda x: x.fillna(0))
    covid["positive_rate"] = covid.groupby("location")["positive_rate"].apply(lambda x: x.fillna(0))
    covid["stringency_index"] = covid.groupby("location")["stringency_index"].apply(lambda x: x.fillna(0))
    covid["population"] = covid.groupby("location")["population"].apply(lambda x: x.ffill().bfill().fillna(0))
    covid["population_density"] = covid.groupby("location")["population_density"].apply(lambda x: x.ffill().bfill().fillna(0))
    covid["median_age"] = covid.groupby("location")["median_age"].apply(lambda x: x.ffill().bfill().fillna(0))
    covid["aged_65_older"] = covid.groupby("location")["aged_65_older"].apply(lambda x: x.ffill().bfill().fillna(0))
    covid["aged_70_older"] = covid.groupby("location")["aged_70_older"].apply(lambda x: x.ffill().bfill().fillna(0))
    covid["gdp_per_capita"] = covid.groupby("location")["gdp_per_capita"].apply(lambda x: x.ffill().bfill().fillna(0))
    covid["extreme_poverty"] = covid.groupby("location")["extreme_poverty"].apply(lambda x: x.ffill().bfill().fillna(0))
    covid["cardiovasc_death_rate"] = covid.groupby("location")["cardiovasc_death_rate"].apply(lambda x: x.ffill().bfill().fillna(0))
    covid["diabetes_prevalence"] = covid.groupby("location")["diabetes_prevalence"].apply(lambda x: x.ffill().bfill().fillna(0))
    covid["female_smokers"] = covid.groupby("location")["female_smokers"].apply(lambda x: x.ffill().bfill().fillna(0))
    covid["male_smokers"] = covid.groupby("location")["male_smokers"].apply(lambda x: x.ffill().bfill().fillna(0))
    covid["handwashing_facilities"] = covid.groupby("location")["handwashing_facilities"].apply(lambda x: x.ffill().bfill().fillna(0))
    covid["hospital_beds_per_thousand"] = covid.groupby("location")["hospital_beds_per_thousand"].apply(lambda x: x.ffill().bfill().fillna(0))
    covid["life_expectancy"] = covid.groupby("location")["life_expectancy"].apply(lambda x: x.ffill().bfill().fillna(0))
    covid["human_development_index"] = covid.groupby("location")["human_development_index"].apply(lambda x: x.ffill().bfill().fillna(0))

    print(covid["new_cases"])

    combined_datasets = pd.merge(covid, school, how = 'left', right_on = ['date', 'iso_code'], left_on = ['date', 'iso_code'])


    return combined_datasets.to_json(date_format='iso', orient='split')

@app.callback(Output('saved_data', 'children'),
              [Input('upload-data', 'contents'), Input('upload-school', 'contents')], prevent_initial_call = True)
def update_output(covid, school):
    if covid is not None and school is not None: 
        return parse_contents(covid, school)
    else:
        raise dash.exceptions.PreventUpdate


@app.callback([Output("predictions_container", "children")],
[Input("saved_data", "children")], prevent_initial_call = True)
def initialize_graphs(jsonified_data):
    if(jsonified_data is not None):
        return add_predictions(jsonified_data)
    else:
        raise dash.exceptions.PreventUpdate


@app.callback([Output("edu_container", "children")],
[Input("saved_data", "children")], prevent_initial_call = True)
def initialize_graphs(jsonified_data):
    if(jsonified_data is not None):
        return add_education(jsonified_data)
    else:
        raise dash.exceptions.PreventUpdate


@app.callback([Output("analyses_container", "children")],
[Input("saved_data", "children")], prevent_initial_call = True)
def initialize_graphs(jsonified_data):
    if(jsonified_data is not None):
        return add_analyses(jsonified_data)
    else:
        raise dash.exceptions.PreventUpdate


@app.callback([Output("centralMap", "children")],
[Input("saved_data", "children")], prevent_initial_call = True)
def initialize_graphs(jsonified_data):
    if(jsonified_data is not None):
        return add_central_map(jsonified_data, "total_cases_per_million")
    else:
        raise dash.exceptions.PreventUpdate


@app.callback([Output("leftSide", "children")],
[Input("saved_data", "children")], prevent_initial_call = True)
def initialize_graphs(jsonified_data):
    if(jsonified_data is not None):
        return add_ranking(jsonified_data, "total_cases_per_million", 10)
    else:
        raise dash.exceptions.PreventUpdate



def get_info(feature=None):
    header = [html.H4("COVID cases per million")]
    if not feature:
        return header + ["Select a country to see more data"]

    quantity =  feature["properties"]["total_cases_per_million"]

    info = ""

    if(quantity is None):
        info = "Data not available"
    else:
        info = str(quantity) + " total cases per million"

    return header + [html.B(feature["properties"]["ADMIN"]), html.Br(),
                   info]


def add_central_map(covid_data, color_col):

    covid_data = pd.read_json(covid_data, orient="split")

    covid_data['date'] = pd.to_datetime(covid_data['date'], dayfirst=True)
    covid_data.sort_values('date', ascending = True, inplace = True)


    graph_divs = []

    ds = covid_data[covid_data["location"] != "World"][color_col]
    max_cases = ds.max()
    twosd_distance = ds.mean() + 4*ds.std()

    colorscale = topmap_colors

    optimal_step = twosd_distance/len(topmap_colors)


    classes = np.arange(start = 0, stop = twosd_distance, step = round(optimal_step, -int(floor(log10(abs(optimal_step))))))[:len(topmap_colors)]
    #colorscale = ['#FFEDA0', '#FED976', '#FEB24C', '#FD8D3C', '#FC4E2A', '#E31A1C', '#BD0026', '#800026']
    style = dict(weight=2, opacity=1, color='white', dashArray='3', fillOpacity=0.7)
    # Create colorbar.
    ctg = ["{}+".format(cls, classes[i + 1]) for i, cls in enumerate(classes[:-1])] + ["{}+".format(classes[-1])]
    colorbar = dlx.categorical_colorbar(categories=ctg, colorscale=colorscale, width=400, height=30, position="bottomleft")

    with urllib.request.urlopen('http://127.0.0.1:8050/static/countries.geojson') as f:
        data = json.load(f)


    for feature in data["features"]:
        feature["properties"][color_col] = covid_data[covid_data["iso_code"] == feature["properties"]["ISO_A3"]][color_col].max()
        feature["properties"]["total_deaths"] = covid_data[covid_data["iso_code"] == feature["properties"]["ISO_A3"]]["total_deaths"].max()
        feature["properties"]["total_cases"] = covid_data[covid_data["iso_code"] == feature["properties"]["ISO_A3"]]["total_cases"].max()


    geojson = dl.GeoJSON(data=data,  # url to geojson file
                        options=dict(style=dlx.choropleth.style),  # how to style each polygon
                        zoomToBounds=True,  # when true, zooms to bounds when data changes (e.g. on load)
                        zoomToBoundsOnClick=True,  # when true, zooms to bounds of feature (e.g. polygon) on click
                        hoverStyle=dict(weight=5, color='#666', dashArray=''),  # special style applied on hover
                        hideout=dict(colorscale=colorscale, classes=classes, style=style, color_prop=color_col),
                        id="geojson")

    # https://dash-leaflet.herokuapp.com/#geojson
    
    # Create info control.
    info = html.Div(children=get_info(), id="info", className="info",
                    style={"position": "absolute", "top": "10px", "right": "10px", "z-index": "1000"})

    map_div = html.Div([dl.Map(children=[dl.TileLayer(), geojson, colorbar, info])],
                      style={'width': '100%', 'height': '400px', 'margin': "auto", "display": "block"}, id="map")


    return [map_div]


def add_ranking(covid_data, ranking_col, k):

    covid_data = pd.read_json(covid_data, orient="split")

    covid_data['date'] = pd.to_datetime(covid_data['date'], dayfirst=True)
    covid_data.sort_values('date', ascending = True, inplace = True)

    covid_last = covid_data[covid_data["date"] == covid_data["date"].max()]
    covid_last.sort_values(ranking_col, ascending = False, inplace = True)

    y =  covid_last["location"].head(k)

    fig = go.Figure(go.Bar(
            x = covid_last[ranking_col].head(k),
            y = y,
            text =  y,
            textposition = "inside",
            orientation='h'),
            layout={
                'margin': {'l': 20, 'r': 0, 't': 5, 'b': 0},
            }
    )


    fig.update_layout(yaxis = dict(categoryorder = 'total ascending'))
    fig.update_layout(yaxis_visible=True, yaxis_showticklabels=False, yaxis_title = "Ranking (total cases/million)")

    return [dcc.Graph(figure = fig, id = "left_ranking")]



@app.callback(Output("info", "children"), [Input("geojson", "hover_feature")], prevent_initial_call = True)
def info_hover(feature):
    return get_info(feature)

@app.callback(Output("rightSide", "children"), [Input("geojson", "click_feature")], prevent_initial_call = True)
def country_click(feature):
    if feature is not None:
        header = [html.H1(feature["properties"]["ADMIN"])]

        compare_deaths = go.Figure(data = [go.Bar(x = ["Cases", "Deaths (x 10)"], y = [feature["properties"]["total_cases"], 10*feature["properties"]["total_deaths"]])])
        compare_deaths.update_layout(margin={"r":5,"t":60,"l":5,"b":5}, title = "Cases and deaths (x 10)")

        # html.Img(src = app.get_asset_url("who.png"))
        #set_location = [html.Button(id = {'type': "set_location", 'index': feature["properties"]["ADMIN"]}, children = [html.I(className="far fa-chart-bar")])]
        return header + [dcc.Graph(figure = compare_deaths, id = "sideHist"), html.Br(), html.A([html.P("Information from WHO")], target = "_blank", href = "https://www.who.int/countries/" + feature["properties"]["ISO_A3"])]
 

def add_predictions(jsonified_data):

    graph_divs = []
    
    # SARIMAX
    axes = []
    axes.append(Axis("Date", 'data["date"]'))
    axes.append(Axis("New cases", ["new_cases_smoothed"], [""]))


    filters = []
    filters.append(Filter(default_value = ["Italy"], column_name = "location", multi = False))

    graph_infos.append(GraphInfo(dataset = jsonified_data,  prediction_method = "SARIMAX", title = "SARIMAX", axes = axes, filters = filters))
        
    #graph_divs.append(new_custom_graph())

    graph_divs.append(predict_world_cases(filters[0].default_value[0], None)[0])


    # Prophet
  

    filters = []
    filters.append(Filter(default_value = ["Italy"], column_name = "location", multi = False))

    graph_infos.append(GraphInfo(dataset = jsonified_data,  prediction_method = "Prophet", title = "Prophet", axes = axes, filters = filters))
        
    graph_divs.append(predict_world_cases(filters[0].default_value[0], None)[0])



    # VAR
  
    
    filters = []
    filters.append(Filter(default_value = ["Italy"], column_name = "location", multi = False))

    graph_infos.append(GraphInfo(dataset = jsonified_data, prediction_method = "VAR", title = "VAR", axes = axes, filters = filters))
        
    graph_divs.append(predict_world_cases(filters[0].default_value[0], None)[0])


    # Initialize preset container and return
    preset_container = html.Div(children = graph_divs)

    return [preset_container]
    
def add_analyses(jsonified_data):
    graph_divs = []

    



    # Graph 3: cumulative tests, confirmed cases, deaths per million people
    axes = []
    axes.append(Axis("Date", 'data["date"]'))
    axes.append(Axis("Tests, cases and deaths per million", ['data["total_tests_per_thousand"]*1000', 'data["total_cases_per_million"]', 'data["total_deaths_per_million"]'], ["Total tests", "Total cases", "Total deaths"], log_scale = True))

    filters = []
    filters.append(Filter(filter_name = "Median age", filter_type="RangeSlider", column_name = "median_age"))

    filters.append(Filter(default_value = ["Norway"], column_name = "location", multi = True))

    graph_infos.append(GraphInfo(dataset = jsonified_data,  title = "Cumulative tests, confirmed cases and deaths per million", axes = axes, filters = filters))
        
    graph_divs.append(new_custom_graph())

    # Graph 1: new cases by population
    axes = []
    axes.append(Axis("Date", 'data["date"]'))
    axes.append(Axis("New cases to population percentage", ['100*data["new_cases"]/data["population"]'], ["New cases by population"]))

    filters = []
    filters.append(Filter(default_value = ["Italy"], column_name = "location", multi = True))

    graph_infos.append(GraphInfo(dataset = jsonified_data,  divide_traces = True, title = "New cases by population", axes = axes, filters = filters))
        
    graph_divs.append(new_custom_graph())

    
    # Graph 1: new cases to deaths
    axes = []
    axes.append(Axis("Date", 'data["date"]'))
    axes.append(Axis("New deaths to cases ratio [%]", ['100*data["new_deaths"].shift(periods = -11)/data["new_cases"]'], ["New deaths to new cases ratio"], log_scale = True))

    filters = []
    filters.append(Filter(default_value = ["Italy"], column_name = "location", multi = True))

    graph_infos.append(GraphInfo(dataset = jsonified_data,  divide_traces = True, title = "New deaths to new cases ratio (11 days shifted)", axes = axes, filters = filters))
        
    graph_divs.append(new_custom_graph())


    
    # Graph 1: new increment with respect to previous (hospitalization)
    axes = []
    axes.append(Axis("Date", 'data["date"]'))
    axes.append(Axis("Hospitalized patients increment", ['data["hosp_patients"].diff()/data["hosp_patients"].shift(periods = 1)'], ["New hospitalizations wrt the previous day"]))

    filters = []
    filters.append(Filter(default_value = ["Italy"], column_name = "location", multi = True))

    graph_infos.append(GraphInfo(dataset = jsonified_data,  divide_traces = True, title = "New hospitalizations wrt the previous day", axes = axes, filters = filters))
        
    graph_divs.append(new_custom_graph())


    
    # Graph 1: new increment with respect to previous (ICUs)
    axes = []
    axes.append(Axis("Date", 'data["date"]'))
    axes.append(Axis("ICU patients increment", ['data["icu_patients"].diff()/data["icu_patients"].shift(periods = 1)'], ["New ICUs wrt the previous day"]))

    filters = []
    filters.append(Filter(default_value = ["Italy"], column_name = "location", multi = True))

    graph_infos.append(GraphInfo(dataset = jsonified_data, divide_traces = True,  title = "New ICUs wrt the previous day", axes = axes, filters = filters))
        
    graph_divs.append(new_custom_graph())



  
    
    
    # 4: Map with deaths per million
    axes = []
    axes.append(Axis("x", 'data["location"]'))
    axes.append(Axis("y", ['data["total_deaths_per_million"]'], ["Total deaths per million"]))

    filters = []
    #filters.append(Filter(default_value = ["2020-10-19"], column_name = "date", multi = True))

    graph_infos.append(GraphInfo(dataset = jsonified_data,  title = "Total deaths per million", axes = axes, filters = filters, animation_frame = 'data["date"].astype(str)'))
        
    graph_divs.append(new_custom_map())

    
    # 5: Map with cases per million
    axes = []
    axes.append(Axis("x", 'data["location"]'))
    axes.append(Axis("y", ['data["total_cases_per_million"]'], ["Total cases per million"]))

    filters = []

    graph_infos.append(GraphInfo(dataset = jsonified_data,  title = "Total cases per million", axes = axes, filters = filters))
        
    graph_divs.append(new_custom_map())
    


    # 5: Deaths to cases ratio
    axes = []
    axes.append(Axis("x", 'data["location"]'))
    axes.append(Axis("y", ['data["total_deaths"]/data["total_cases"]'], ["Deaths to cases ratio"]))

    filters = []
    graph_infos.append(GraphInfo(dataset = jsonified_data,  title = "Deaths to cases ratio", axes = axes, filters = filters, animation_frame = 'data["date"].astype(str)'))
        
    graph_divs.append(new_custom_map())
    

  
 



    # Stringency and cases per million
    axes = []
    axes.append(Axis("Date", 'data["date"]'))
    axes.append(Axis("DSR", ['data["new_deaths_per_million"]/data["stringency_index"]'], ["DSR"]))

    filters = []

    filters.append(Filter(default_value = ["Norway"], column_name = "location", multi = True))

    graph_infos.append(GraphInfo(dataset = jsonified_data,  divide_traces = True, title = "Deaths to stringency ratio (DSR)", axes = axes, filters = filters))
        
    graph_divs.append(new_custom_graph())

   

    
    # Deaths / Stringency (1 week shift)
    axes = []
    axes.append(Axis("Date", 'data["date"]'))
    axes.append(Axis("DSR", ['data["new_deaths_per_million"]/(1 + data["stringency_index"].shift(periods = 7))'], ["DSR (1 week shift)"]))

    filters = []
    filters.append(Filter(default_value = ["Italy"], column_name = "location", multi = True))

    graph_infos.append(GraphInfo(dataset = jsonified_data,  divide_traces = True, title = "Death to stringency ratio (1 week shift)", axes = axes, filters = filters))
        
    graph_divs.append(new_custom_graph())






    


    # Graph 1: new increment with respect to previous (hospitalization)
    axes = []
    axes.append(Axis("Date", 'data["date"]'))
    axes.append(Axis("Hospitalized patients increment", ['data["hosp_patients"].diff()/data["hosp_patients"].shift(periods = 1)'], ["New hospitalizations wrt the previous day"]))

    filters = []
    filters.append(Filter(default_value = ["Italy"], column_name = "location", multi = True))

    graph_infos.append(GraphInfo(dataset = jsonified_data,  divide_traces = True, title = "New hospitalizations wrt the previous day", axes = axes, filters = filters))
        
    graph_divs.append(new_custom_graph())


    # Population density vs cases per million
    axes = []
    axes.append(Axis("Population density", 'mask_max(data, "total_cases_per_million")["population_density"]'))
    axes.append(Axis("Cases per million", ['[data["total_cases_per_million"].max()]'], ["Cases per million"]))

    filters = []
    filters.append(Filter(show_on_marker = True, default_value = ["Norway"], column_name = "location", multi = True))

    graph_infos.append(GraphInfo(dataset = jsonified_data,  title = "Population density vs cases per million", axes = axes, filters = filters, hide_side_legend = True, plot_type = "markers+text", same_color = True))
        
    graph_divs.append(new_custom_graph())


    # 5: New cases per area
    axes = []
    axes.append(Axis("x", 'data["location"]'))
    axes.append(Axis("y", ['data["new_cases"]*data["population_density"]/data["population"]'], ["Cases per area"]))

    filters = []
    graph_infos.append(GraphInfo(dataset = jsonified_data,  std = 1, min_animation = 0, title = "Cases per square kilometer", axes = axes, filters = filters, animation_frame = 'data["date"].astype(str)'))
        
    graph_divs.append(new_custom_map())
    


    # Multi-filter
    axes = []
    axes.append(Axis("Date", 'data["date"]'))
    axes.append(Axis("Total cases per million", ['data["total_cases_per_million"]'], ["Cases per million"]))

    filters = []
    filters.append(Filter(filter_name = "Median age", filter_type="RangeSlider", column_name = "median_age"))
    filters.append(Filter(filter_name = "Population density", filter_type="RangeSlider", column_name = "population_density", n_steps = 100))
    filters.append(Filter(filter_name = "GDP per capita", filter_type="RangeSlider", column_name = "gdp_per_capita", n_steps = 20))

    filters.append(Filter(filter_name = "Population", filter_type="RangeSlider", column_name = "population", n_steps = 40))
    filters.append(Filter(filter_name = "Life expectancy", filter_type="RangeSlider", column_name = "life_expectancy", n_steps = 100))
    filters.append(Filter(filter_name = "Human development index", filter_type="RangeSlider", column_name = "human_development_index", n_steps = 100, prec = 2))
    
    filters.append(Filter(default_value = ["Norway"], column_name = "location", multi = True))

    graph_infos.append(GraphInfo(dataset = jsonified_data,  divide_traces = True, title = "Cases per million (with multiple filters)", axes = axes, filters = filters))
        
    graph_divs.append(new_custom_graph())






    # Initialize preset container and return
    preset_container = html.Div(children = graph_divs)
    return [preset_container]

def add_education(jsonified_data):
    graph_divs = []

    # School open vs stringency
    axes = []
    axes.append(Axis("Date", 'data["date"]'))
    axes.append(Axis("ESCO", ['data["stringency_index"]-100*(data["Physical_education"]-1)'], ["ESCO"]))

    filters = []
    filters.append(Filter(default_value = ["Norway"], column_name = "location", multi = True))

    graph_infos.append(GraphInfo(dataset = jsonified_data,  divide_traces = True, title = "Education stringency coherency (ESCO)", axes = axes, filters = filters))
        
    graph_divs.append(new_custom_graph())


    # Days of opening
    axes = []
    axes.append(Axis("Country", 'data["location"]'))
    axes.append(Axis("Number of days", ['[data[data["Status"] == "Fully open"]["Status"].count()]', '[data[data["Status"] == "Partially open"]["Status"].count()]', '[data[data["Status"] == "Closed due to COVID-19"]["Status"].count()]'], ["Fully open", "Partially open", "Closed due to COVID-19"]))

    filters = []
    filters.append(Filter(filter_type="DatePickerRange", column_name = "date"))
    filters.append(Filter(default_value = ["Norway"], column_name = "location", multi = True))

    graph_infos.append(GraphInfo(dataset = jsonified_data,  go_type = "Bar", title = "School status", axes = axes, filters = filters))
        
    graph_divs.append(new_custom_graph())


    # School stringency
    axes = []
    axes.append(Axis("Date", 'data["date"]'))
    axes.append(Axis("School stringency", ['data["Physical_education"]'], ["School stringency"]))

    filters = []
    filters.append(Filter(default_value = ["Norway"], column_name = "location", multi = True))

    graph_infos.append(GraphInfo(dataset = jsonified_data,  divide_traces = True, title = "Closing status (1 for fully open, 2 for fully closed)", axes = axes, filters = filters))
        
    graph_divs.append(new_custom_graph())



    # Initialize preset container and return
    preset_container = html.Div(children = graph_divs)
    return [preset_container]

def new_custom_map():
    ind = len(graph_infos)-1
    graph_info = graph_infos[ind]
    covid_data = pd.read_json(graph_info.dataset, orient="split")

    covid_data['date'] = pd.to_datetime(covid_data['date'], dayfirst=True)
    covid_data.sort_values('date', ascending = True, inplace = True)

    #map_div = dash_draggable.dash_draggable(axis="both", grid=[30, 30], children = [])
    map_div = html.Div(children = [])

    data = covid_data

    subindex = 0
    for f in graph_info.filters:
        dropdown_menu = dcc.Dropdown(id={'type': 'DD', 'index': ind, 'internal_index': subindex}, value=f.default_value, options = [{'label': i, 'value': i} for i in covid_data[f.column_name].unique()], multi=f.multi)
        map_div.children.append(dropdown_menu)
        data = covid_data

        for defv in f.default_value:
            data = data[data[f.column_name] == defv]

        subindex += 1

    color_data = eval(graph_info.axes[1].content[0])
    min_color = max(color_data.min(), color_data.mean() - graph_info.std*color_data.std())
    max_color = min(color_data.max(), color_data.mean() + graph_info.std*color_data.std())
   
    if(graph_info.min_animation is not None):
        min_color = max(min_color, graph_info.min_animation)

    if(graph_info.max_animation is not None):
        max_color = min(max_color, graph_info.max_animation)


    fig = px.choropleth(data, range_color = [min_color, max_color], labels = dict(animation_frame = 'Day'), animation_frame = eval(graph_info.animation_frame), locations = eval(graph_info.axes[0].content).values, color = color_data.values, color_continuous_scale = graph_info.colorscale, locationmode = graph_info.location_mode)
    
    fig.update_layout(margin={"r":5,"t":60,"l":5,"b":5}, title = graph_info.title)
    fig.update_layout(coloraxis_colorbar=dict(title="Value"))

    graph = dcc.Graph(id={'type': 'MA', 'index': ind}, figure=fig)
    graph.className = "graph_div graph map"

    resize_button = html.I("")
    resize_button.className = "resizeGraph fas fa-expand-alt"

    move_button = html.I("")
    move_button.className = "moveGraph fas fa-arrows-alt"

    #map_div.children.append(move_button)
    #map_div.children.append(resize_button)
    map_div.children.append(graph)

    return map_div

def new_custom_graph():
    ind = len(graph_infos)-1
    graph_info = graph_infos[ind]
    covid_data = pd.read_json(graph_info.dataset, orient="split")

    #graph_div = dash_draggable.dash_draggable(axis="both", grid=[30, 30], children = [])
    graph_div = html.Div(children = [])
    #fig = px.line(title = graph_info.title)

    fig = go.Figure(layout = {'title': graph_info.title})


    subindex = 0
    for f in graph_info.filters:

        data = covid_data


        filter_menu = None
        if(f.filter_type == "Dropdown"):
            filter_menu = dcc.Dropdown(id={'type': 'DD', 'index': ind, 'internal_index': subindex}, value=f.default_value, options = [{'label': i, 'value': i} for i in covid_data[f.column_name].unique()], multi=f.multi)
        elif(f.filter_type == "DatePickerRange"):
            filter_menu = dcc.DatePickerRange(id={'type': 'DP', 'index': ind, 'internal_index': subindex})
        elif(f.filter_type == "RangeSlider"):
            optimal_step = 1

            max_range = ceil(data[f.column_name].max()*1.1)
            min_range = floor(data[f.column_name].min()*0.9)
            n_steps = f.n_steps


            if(n_steps is not None and f.prec != 0):

                prec_pow = pow(10, f.prec)

                optimal_step = max(1/prec_pow, (floor(prec_pow*(max_range - min_range))/prec_pow)/n_steps)

            elif(n_steps is not None and f.prec == 0):
                optimal_step = ceil(max(1, (floor(max_range - min_range))/n_steps))


            filter_menu = dcc.RangeSlider(tooltip = {'always_visible': True, 'placement': 'bottom'}, dots = True, id =  {'type': 'SR', 'index': ind, 'internal_index': subindex}, min=min_range, max=max_range, step=optimal_step, value=[floor(data[f.column_name].min()), ceil(data[f.column_name].max())])
        
        filter_name = f.filter_name

        filter_div = html.Div(className = "filterDiv", children = [])

        if(filter_name != ""):
            filter_div.children.append(html.Div(children = [filter_name]))
        
        filter_div.children.append(filter_menu)
        graph_div.children.append(filter_div)

        if(f.filter_type == "Dropdown"):
        
            def_val_list = f.default_value

            if(isinstance(f.default_value, str)):
                def_val_list = eval(f.default_value)


            number_filters = len(def_val_list)
            num_cols = ceil(sqrt(number_filters))
            num_rows = ceil(number_filters/num_cols)

            plot_titles = []

            for val in def_val_list:
                plot_titles.append(val)


            if(graph_info.divide_traces is True):
                fig = make_subplots(rows = num_rows, cols = num_cols, subplot_titles = plot_titles)
                fig.update_layout(title = graph_info.title,  yaxis = dict(title=graph_info.axes[1].label), xaxis = dict(title=graph_info.axes[0].label))



            for defv in def_val_list:
                data = covid_data.loc[covid_data[f.column_name] == defv]

            for lab in range(0, len(graph_info.axes[1].content)):
                y_trace = graph_info.axes[1].content[lab]
                label = graph_info.axes[1].labels[lab]
                if(len(f.default_value) > 1):
                    if(f.show_on_marker is True):
                        label = defv
                    else:
                        if(len(graph_info.filters) > 1):
                            label = filter_value[i][j] + " - " + label
                        else:
                            label = filter_value[i][j]

                if(graph_info.divide_traces is False):
                    if(graph_info.go_type == "Scatter"):      
                        fig.add_trace(go.Scatter(mode = graph_info.plot_type, name=label, text = label, textposition = 'top center', x=eval(graph_info.axes[0].content), y=eval(y_trace)))
                    elif(graph_info.go_type == "Bar"):
                        fig.add_trace(go.Bar(name=label, x=eval(graph_info.axes[0].content), y=eval(y_trace)))
                else:
                    if(graph_info.go_type == "Scatter"):      
                        fig.add_trace(go.Scatter(mode = graph_info.plot_type, name=label, text = label, textposition = 'top center', x=eval(graph_info.axes[0].content), y=eval(y_trace)), col = 1 + lab%num_cols, row = 1 + floor(lab/num_rows))
                    elif(graph_info.go_type == "Bar"):
                        fig.add_trace(go.Bar(name=label, x=eval(graph_info.axes[0].content), y=eval(y_trace)), col = 1 + lab%num_cols, row = 1 + floor(lab/num_rows))

        subindex += 1

        covid_data = data
        
    fig.update_layout(height = graph_info.height, width = graph_info.width, showlegend=not graph_info.hide_side_legend, yaxis=dict(title=graph_info.axes[1].label), xaxis=dict(title=graph_info.axes[0].label))

    if(graph_info.axes[0].log_scale is True):
        fig.update_xaxes(type="log")

    if(graph_info.axes[1].log_scale is True):
        fig.update_yaxes(type="log")

    graph = dcc.Graph(id={'type': 'GR', 'index': ind}, figure=fig)
    graph.className = "graph_div graph"

    resize_button = html.I("")
    resize_button.className = "resizeGraph fas fa-expand-alt"

    move_button = html.I("")
    move_button.className = "moveGraph fas fa-arrows-alt"

    #graph_div.children.append(move_button)
    #graph_div.children.append(resize_button)
    graph_div.children.append(graph)


    return graph_div





@app.callback( 
Output({'type': 'GRPR', 'index': MATCH}, 'figure'), 
    [Input({'type':'PREDD', 'index': MATCH, 'internal_index': ALL}, 'value'),
    Input({'type':'PREDD', 'index': MATCH, 'internal_index': ALL}, 'id'),
    Input("saved_data", "children")], prevent_initial_call = True
    )
def update_predictions(filter_value, filter_id, jsonified_data):
    return predict_world_cases(filter_value, filter_id, jsonified_data)[1]


def predict_world_cases(filter_value, filter_id, jsonified_data = None):

    ind = None
    intind = None

    if jsonified_data is not None:
        if(len(filter_id) > 0):
            ind = filter_id[0]['index']
            intind = filter_id[0]['internal_index']

    else:
         ind = len(graph_infos)-1
         
    graph_info = graph_infos[ind]


    # Collect the graph information
   
    prediction_method = graph_info.prediction_method
    quantity_to_predict = graph_info.axes[1].content[0]


    if(jsonified_data is None):
        covid_data = pd.read_json(graph_info.dataset, orient="split")
    else:
        covid_data = pd.read_json(jsonified_data, orient="split")

    f = graph_info.filters[0]

    filter_menu = dcc.Dropdown(id={'type': 'PREDD', 'index': ind, 'internal_index': 0}, value=f.default_value[0], options = [{'label': i, 'value': i} for i in covid_data[f.column_name].unique()], multi=f.multi)

    filter_div = html.Div(className = "filterDiv", children = [])
        
    filter_div.children.append(filter_menu)

    #graph_div = dash_draggable.dash_draggable(axis="both", grid=[30, 30], children = [])
    graph_div = html.Div(children = [])
    fig = go.Figure(layout = {'title': graph_info.title, 'xaxis_title': "Date", 'yaxis_title': "New cases (smoothed)"})

    first_phase_start = datetime.datetime.strptime("2020-01-22", "%Y-%m-%d")

    # Consider the dataset only from when there are meaningful collected data
    dataset = covid_data.loc[covid_data['date'] >= first_phase_start]
    
    if(not isinstance(filter_value, str)):
        filter_value = filter_value[0]

    var_data = covid_data.loc[covid_data["location"] == filter_value]
    var_data.set_index("date", inplace = True, drop = True)
    var_data.index = var_data.index.to_period("D")
    var_data = var_data.sort_index()
    population = var_data["population"].iloc[0]
    var_data = var_data.loc[:, (var_data != var_data.iloc[0]).any()] # Delete constant values (useless for predictions)


    try:
        var_data.drop(["tests_units"], axis = 1, inplace = True) # Delete machine-unreadable values (non-standard strings)
    except:
        pass

    try:
        var_data.drop(["Country"], axis = 1, inplace = True) # Delete machine-unreadable values (non-standard strings)
    except:
        pass
        
    try:
        var_data.drop(["Status"], axis = 1, inplace = True) # Delete machine-unreadable values (non-standard strings)
    except:
        pass
        
    try:
        var_data.drop(["Note"], axis = 1, inplace = True) # Delete machine-unreadable values (non-standard strings)
    except:
        pass

    var_data["total_cases"].fillna(method = 'bfill', inplace = True) # it does not make sense to put them at zero, since they are cumulative
    var_data.fillna(0, inplace = True)
       
    cols = var_data.columns



    '''
    derivative = var_data[quantity_to_predict].diff()
    #second_derivative = derivative.diff()
                   
    threshold = 0.1*derivative.max()

    if(prediction_method == "VAR"):
        threshold = 0.02*derivative.max()
 
    
    second_phase_extremes = derivative.loc[derivative < threshold]
    second_phase_extremes = second_phase_extremes.loc[derivative >= 0]
    
    if(len(second_phase_extremes) == 0):
        second_phase_extremes = [0.0]

    second_phase_nomax = second_phase_extremes.loc[second_phase_extremes >= 0]

    second_phase_index = second_phase_nomax.iloc[[-1]].index
   
    print(second_phase_index[0])
    '''

    peaks, properties = scipy.signal.find_peaks(var_data[quantity_to_predict], plateau_size = [0, 50], distance = 30)
    _, beginning_peak, end_peak = scipy.signal.peak_prominences(var_data[quantity_to_predict], peaks)
    

    after_last_beginning = var_data[int(beginning_peak.max()):]

    second_phase_start = after_last_beginning[quantity_to_predict].idxmax(axis = 0, skipna = True)

    
    if(prediction_method == "VAR"):
        second_phase_start = after_last_beginning[quantity_to_predict].idxmin(axis = 0, skipna = True)
        second_phase_start += datetime.timedelta(days = int(len(after_last_beginning)*0.15))

    second_phase_start = datetime.datetime.strptime(str(second_phase_start), "%Y-%m-%d")
    second_phase_start += datetime.timedelta(days = 2)

    #second_phase_start = first_phase_start + datetime.timedelta(days = int(beginning_peak.max())) + datetime.timedelta(days = int(max_index))

    end_available_data = datetime.datetime.strptime(str(pd.Series(var_data.index.to_timestamp().values).iloc[-1]).split(" ")[0] , "%Y-%m-%d")

    train_proportion = abs((second_phase_start - first_phase_start).days)/abs((first_phase_start - end_available_data).days)

    train_proportion = min(train_proportion, 0.99)

    train = var_data[:int(train_proportion*len(var_data))]
    valid = var_data[int(train_proportion*len(var_data)):]
    
    start_date =  str(train.index[-1])
    end_date = str(datetime.datetime.strptime(str(valid.index[-1]), "%Y-%m-%d") + datetime.timedelta(days = 180))

  
    if(prediction_method == "SARIMAX"):

        
        #exog_train = eval(graph_info.additional_columns.replace("exog_var", "train"))
        train = train[quantity_to_predict]
        #exog_test = eval(graph_info.additional_columns.replace("exog_var", "valid"))
        test = valid[quantity_to_predict]

        '''
        print("EXOG 1")
        print(exog_test)
        '''

        step_wise = pm.auto_arima(train, start_p = 1, start_q = 1, test = 'adf', max_p = 5, max_q = 5, m=1, d=2, seasonal=False, start_P = 0, D=0, trace = True, error_action = 'ignore', suppress_warning=True, stepwise = True)

        model = SARIMAX(train, order=step_wise.order) 
    
        # New cases per day
        # World (7, 1, 8), Italy (1, 2, 0), Norway (2, 2, 3)

        results = model.fit(low_memory = True, disp = False)

        '''
        for i in range(180):
            exog_test = exog_test.append(pd.Series(), ignore_index=True)

        exog_test.fillna(method = "ffill", inplace = True)

        print("EXOG 2")
        print(exog_test)
        '''

        #sarimax_prediction = results.predict(exog = big_test, start = start_date, end= end_date)
        sarimax_prediction = results.forecast(steps = len(valid) + 180)

        sarimax_prediction[sarimax_prediction > population] = population
        sarimax_prediction[sarimax_prediction < 0] = 0

        fig.add_trace(go.Scatter(mode = graph_info.plot_type, name="Prediction",  line = dict(dash = "dash"), x=pd.Series(sarimax_prediction.index.to_timestamp().values), y=sarimax_prediction))
    elif(prediction_method == "Prophet"):
        train["ds"] = train.index.to_timestamp()
        train.rename(columns = {'date': 'ds', quantity_to_predict: 'y'}, inplace = True)
        valid.rename(columns = {'date': 'ds', quantity_to_predict: 'y'}, inplace = True)

        m = Prophet()
        # https://stackoverflow.com/questions/54544285/is-it-possible-to-do-multivariate-multi-step-forecasting-using-fb-prophet to add parameters

        m.fit(train)
        
        future = m.make_future_dataframe(periods = 180 + len(valid))

        forecast = m.predict(future)


        forecast[["yhat", "yhat_lower", "yhat_upper"]] = forecast[["yhat", "yhat_lower", "yhat_upper"]].apply(lambda x: [max(0, min(y, population)) for y in x])

        fig.add_trace(go.Scatter(mode = graph_info.plot_type, line_color = "blue",  line = dict(dash = "dash"), name="Prediction", x=forecast["ds"], y=forecast["yhat"]))
        fig.add_trace(go.Scatter(mode = graph_info.plot_type, fill = "tonexty",  marker=dict(color="#989898"), showlegend = False, line = dict(width = 0), fillcolor='rgba(152, 152, 152, 0.5)', name="Prediction (lower bound)", x=forecast["ds"], y=forecast["yhat_lower"]))
        fig.add_trace(go.Scatter(mode = graph_info.plot_type, fill = "tonexty", name="Prediction (upper bound)", marker = dict(color="#989898"), line = dict(width = 0), showlegend = False, x=forecast["ds"], y=forecast["yhat_upper"]))
    elif(prediction_method == "VAR"):

        model = VAR(endog = train)
        model_fit = model.fit()

        prediction = model_fit.forecast(model_fit.y, steps = 180 + len(valid)) # 180 days = 6 months

        #converting predictions to dataframe
        pred = pd.DataFrame(index=range(0,len(prediction)),columns=cols)
        for j in range(0, len(cols)):
            for i in range(0, len(prediction)):
               pred.iloc[i][j] = prediction[i][j]

        future_dates = pd.date_range(start = start_date, periods = 180 + len(valid))

        # pd.Series(valid.index.to_timestamp().values)

        pred[pred[quantity_to_predict] > population] = population
        pred[pred[quantity_to_predict] < 0] = 0
        fig.add_trace(go.Scatter(mode = graph_info.plot_type, name="Prediction", line = dict(dash = "dash"), x = future_dates, y=pred[quantity_to_predict]))
   

    fig.add_trace(go.Scatter(mode = graph_info.plot_type, name="Observation", line_color = "red", x=pd.Series(var_data.index.to_timestamp().values), y=var_data[quantity_to_predict]))

    fig.update_layout(title = prediction_method + " (" + filter_value + ")")

    graph = dcc.Graph(id={'type': 'GRPR', 'index': ind}, figure=fig)
    graph.className = "graph_div graph"

    resize_button = html.I("")
    resize_button.className = "resizeGraph fas fa-expand-alt"

    move_button = html.I("")
    move_button.className = "moveGraph fas fa-arrows-alt"

    #graph_div.children.append(move_button)
    #graph_div.children.append(resize_button)
    graph_div.children.append(filter_div)
    graph_div.children.append(graph)

    return [graph_div, fig]

def rss(y, y_hat):
    return np.square(y - y_hat).sum()
    # Note: the time used by this combination of functions is the same as np.power(y-y_hat, 2).sum() and np.sum(np.square(y-y_hat)), which in my case ranges between 1 and 2 ms
    pass 

def multivarRSS(y, x, w0, w1):
    observed = y
    
    predicted = x.dot(w1[0]) + w0[0]
 
    multivarRSS = rss(observed, predicted)

    return multivarRSS


@app.callback( 
Output({'type': 'GR', 'index': MATCH}, 'figure'), 
    [Input({'type':'DD', 'index': MATCH, 'internal_index': ALL}, 'value'),
    Input({'type':'DD', 'index': MATCH, 'internal_index': ALL}, 'id'),
    
    Input({'type':'DP', 'index': MATCH, 'internal_index': ALL}, 'start_date'),
     Input({'type':'DP', 'index': MATCH, 'internal_index': ALL}, 'end_date'),

    Input({'type':'DP', 'index': MATCH, 'internal_index': ALL}, 'id'),

    Input({'type':'SR', 'index': MATCH, 'internal_index': ALL}, 'id'),
     Input({'type':'SR', 'index': MATCH, 'internal_index': ALL}, 'value'),

     Input("color_blind", "on"),
    Input("saved_data", "children")], prevent_initial_call = True
    )
    
def update_graph(filter_value, filter_id, start_date, end_date, date_id, slider_id, slider_value, color_blind, jsonified_data):
    ind = None
    type_filter = None

    if(len(filter_id) > 0):
        ind = filter_id[0]['index']
        intind = filter_id[0]['internal_index']
    elif(len(date_id) > 0):
        ind = date_id[0]['index']
        intind = date_id[0]['internal_index']
    elif(len(date_id) > 0):
        ind = slider_id[0]['index']
        intind = slider_id[0]['internal_index']
    else:
        ind = None
        intind = None


    graph_info = graph_infos[ind]
    covid_data = pd.read_json(jsonified_data, orient="split")

    data = covid_data

    #fig = px.line(title = graph_info.title)
    fig = go.Figure(layout = {'title': graph_info.title,  'yaxis': dict(title=graph_info.axes[1].label), 'xaxis': dict(title=graph_info.axes[0].label)})
    #fig = go.Figure(layout = {'title': graph_info.title})


    dd_filters = [f for f in graph_info.filters if f.filter_type=="Dropdown"]
    dp_filters = [f for f in graph_info.filters if f.filter_type=="DatePickerRange"]
    sr_filters = [f for f in graph_info.filters if f.filter_type=="RangeSlider"]

    if(covid_data is not None):
        for i in range(0, len(date_id)):
            if(date_id[i]['type'] == "DP"):
                f = dp_filters[i]
                if(start_date[i] is not None and end_date[i] is not None):
                    covid_data = covid_data[covid_data[f.column_name] <= end_date[i]]
                    covid_data = covid_data[covid_data[f.column_name] >= start_date[i]]


        for i in range(0, len(slider_id)):
            if(slider_id[i]['type'] == "SR"):
                f = sr_filters[i]
                if(slider_value[i][0] is not None and slider_value[i][1] is not None):
                    covid_data = covid_data[covid_data[f.column_name] <= slider_value[i][1]]
                    covid_data = covid_data[covid_data[f.column_name] >= slider_value[i][0]]

        for i in range(0, len(filter_id)):
            if(filter_id[i]['type'] == "DD"):

                number_filters = len(filter_value[i])
                num_cols = ceil(sqrt(number_filters))
                num_rows = ceil(number_filters/num_cols)

                
                plot_titles = []

                for val in filter_value[i]:
                    plot_titles.append(val)


                if(graph_info.divide_traces is True):
                    fig = make_subplots(rows = num_rows, cols = num_cols, subplot_titles = plot_titles)  
                    fig.update_layout(title = graph_info.title)



                for j in range(0, len(filter_value[i])): # For each choice in a single dropdown (e.g. list of countries)
                    f = dd_filters[i]
                    data = covid_data[covid_data[f.column_name] == filter_value[i][j]]
                
                    for lab in range(0, len(graph_info.axes[1].content)):

                        for y_trace in graph_info.axes[1].content:
                            y_trace = graph_info.axes[1].content[lab]
                            label = graph_info.axes[1].labels[lab]
                            if(len(filter_id[i]) > 1):
                                label = filter_value[i][j] + " - " + label
                            else:
                                label = filter_value[i][j]

                            if(f.show_on_marker is True):
                                label = filter_value[i][j]
                    
                        #red_component = 20 + 190*(ord(label[0].upper()) - 65)/25
                        #green_component = (20 + 190*(ord(label[1].upper()) - 65)/25)
                        #blue_component = (20 + 190*(ord(label[2].upper()) - 65)/25) + lab*30

                        if(color_blind is True):
                            rgb_color = colorblind_colors[j%3].lstrip("rgb(").rstrip(")").split(",")
                        else:
                            rgb_color = plotly.colors.DEFAULT_PLOTLY_COLORS[j%10].lstrip("rgb(").rstrip(")").split(", ")

                        rgb_color = list(map(int, rgb_color))

                        if(graph_info.same_color is True):
                            computed_color = plotly.colors.DEFAULT_PLOTLY_COLORS[0]
                        else:
                            computed_color = "rgb(" + str(rgb_color[0]*(1+lab/4)) + ", " + str(rgb_color[1]*(1+lab/4)) + ", " + str(rgb_color[2]*(1+lab/4)) + ")"


                        color_properties = graph_info.plot_type[:-1] + "=dict(color = '" + computed_color + "')"
                        
                        
                        if(graph_info.divide_traces is False):
                            if(graph_info.go_type == "Scatter"):      
                                fig.add_trace(go.Scatter(line = dict(color = computed_color),  mode = graph_info.plot_type, name=label, text = label, textposition = 'top center', x=eval(graph_info.axes[0].content), y=eval(y_trace)))
                            elif(graph_info.go_type == "Bar"):
                                fig.add_trace(go.Bar(marker_color = computed_color, name=label, x=eval(graph_info.axes[0].content), y=eval(y_trace)))
                        else:
                            if(graph_info.go_type == "Scatter"):      
                                fig.add_trace(go.Scatter(line = dict(color = computed_color), mode = graph_info.plot_type, name=label, text = label, textposition = 'top center', x=eval(graph_info.axes[0].content), y=eval(y_trace)), col = 1 + j%num_cols, row = 1 + floor(j/num_cols))
                            elif(graph_info.go_type == "Bar"):
                                fig.add_trace(go.Bar(marker_color = computed_color, name=label, x=eval(graph_info.axes[0].content), y=eval(y_trace)), col = 1 + j%num_cols, row = 1 + floor(j/num_cols))
                            
                            if(1 + j%num_cols == 1): 
                                fig.update_yaxes(title_text = graph_info.axes[1].label, col = 1, row = 1 + floor(j/num_cols))
                            if(1 + floor(j/num_cols) == num_rows):
                                fig.update_xaxes(title_text = graph_info.axes[0].label, row = num_rows, col = 1 + j%num_cols)
                            if(sqrt(number_filters) != floor(sqrt(number_filters)) and 1 + floor(j/num_cols) == num_rows - 1 and number_filters%num_cols <= j%num_cols):
                                fig.update_xaxes(title_text = graph_info.axes[0].label, row = num_rows - 1, col = 1 + j%num_cols)



        fig.update_layout(showlegend=not graph_info.hide_side_legend)
        fig.update_xaxes(matches='x')
        fig.update_yaxes(matches='y')

        if(graph_info.axes[0].log_scale is True):
            fig.update_xaxes(type="log")

        if(graph_info.axes[1].log_scale is True):
            fig.update_yaxes(type="log")
        
        return fig
    else:
        raise dash.exceptions.PreventUpdate




if __name__ == '__main__':
    app.run_server(debug=True)
