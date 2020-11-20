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
from math import log10, floor, ceil


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

covid_data = None
dimensions = []
graph_infos = []

import urllib.request




external_stylesheets = ["static/style.css", "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.14.0/css/all.min.css"]
external_scripts = ["static/moveGraphs.js", "static/handleMenu.js"]

app = dash.Dash(__name__, external_stylesheets=external_stylesheets, external_scripts=external_scripts)



class GraphInfo:
    def __init__(self, dataset, title, graph_type = "line", axes = [], color = None, filters = [], animation = None, map_type = 'choropleth', location_mode = 'country names', colorscale='Portland', animation_frame = 'data["date"].astype(str)', min_animation = 0, max_animation = 1, plot_type = "lines", hide_side_legend = False, width = 650, height = 500):
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

    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True, indent=4)

class Axis:
    def __init__(self, label = "", content = [], labels = [], log_scale = False):
        self.label = label
        self.log_scale = log_scale
        self.content = content
        self.labels = labels

class Filter:
    def __init__(self, column_name = "", default_value = [], multi = True, filter_type="Dropdown", start_date = "2020-01-01", end_date="2020-12-31", show_on_marker = False, filter_name = ""):
        self.column_name = column_name
        self.default_value = default_value
        self.multi = multi
        self.filter_type = filter_type
        self.start_date = start_date
        self.end_date = end_date
        self.show_on_marker = show_on_marker
        self.filter_name = filter_name

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
        html.H1(children=[html.I("", className="fas fa-virus"), html.Span("   "), html.Span(project_name)]),
       
        html.A(
            id='add-graph',
            children=html.Div([
                html.I("", className="fas fa-plus")
            ])
        ),
        ]),

        html.Div(id = "leftSide", children = []),
        html.Div(id = "centralMap", children = []),
        html.Div(id = "rightSide", children = []),

        html.Ul(id = "three_buttons", children = [html.Li(id = "general_button", className="currentlySelected", children = ["General analyses"]), html.Li(id = "edu_button", children = ["Education"]), html.Li(id = "pred_button", children = ["Predictions"])]),
        html.Div(id="predictions_container", className = "graphs_container", children=[]),
        html.Div(id="edu_container", className = "graphs_container", children=[]),
        html.Div(id="analyses_container", className = "graphs_container", children=[]),

        html.Div(id="filter_equivalence", style={'display': 'none'}),

        # Hidden div inside the app that stores the intermediate value
        html.Div(id='saved_data', style={'display': 'none'}),
        html.Div(id='saved_school', style={'display': 'none'})

    ],
    )

app.layout = generate_layout

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

    school["Physical_education"] = pd.Series()
    school.loc[school["Status"] == "Fully open", "Physical_education"] = 2 
    school.loc[school["Status"] == "Partially open", "Physical_education"] = 1.5
    school.loc[school["Status"] == "Closed due to COVID-19", "Physical_education"] = 1


    combined_datasets = pd.merge(covid, school, how = 'left', right_on = ['date', 'iso_code'], left_on = ['date', 'iso_code'])
    print(combined_datasets[combined_datasets["location"] == "Italy"])

    return combined_datasets.to_json(date_format='iso', orient='split')

@app.callback(Output('saved_data', 'children'),
              [Input('upload-data', 'contents'), Input('upload-school', 'contents')])
def update_output(covid, school):
    if covid is not None and school is not None:
        return parse_contents(covid, school)
    else:
        raise dash.exceptions.PreventUpdate


@app.callback([Output("predictions_container", "children")],
[Input("saved_data", "children")])
def initialize_graphs(jsonified_data):
    if(jsonified_data is not None):
        return add_predictions(jsonified_data)
    else:
        raise dash.exceptions.PreventUpdate


@app.callback([Output("edu_container", "children")],
[Input("saved_data", "children")])
def initialize_graphs(jsonified_data):
    if(jsonified_data is not None):
        return add_education(jsonified_data)
    else:
        raise dash.exceptions.PreventUpdate


@app.callback([Output("analyses_container", "children")],
[Input("saved_data", "children")])
def initialize_graphs(jsonified_data):
    if(jsonified_data is not None):
        return add_analyses(jsonified_data)
    else:
        raise dash.exceptions.PreventUpdate


@app.callback([Output("centralMap", "children")],
[Input("saved_data", "children")])
def initialize_graphs(jsonified_data):
    if(jsonified_data is not None):
        return add_central_map(jsonified_data, "total_cases_per_million")
    else:
        raise dash.exceptions.PreventUpdate


@app.callback([Output("leftSide", "children")],
[Input("saved_data", "children")])
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

    max_cases = covid_data[covid_data["location"] != "World"][color_col].max()
    optimal_step = max_cases/5

    print(max_cases)

    classes = np.arange(start = 0, stop = max_cases, step = round(optimal_step, -int(floor(log10(abs(optimal_step))))))
    
    colorscale = ['#FFEDA0', '#FED976', '#FEB24C', '#FD8D3C', '#FC4E2A', '#E31A1C', '#BD0026', '#800026']
    style = dict(weight=2, opacity=1, color='white', dashArray='3', fillOpacity=0.7)
    # Create colorbar.
    ctg = ["{}+".format(cls, classes[i + 1]) for i, cls in enumerate(classes[:-1])] + ["{}+".format(classes[-1])]
    colorbar = dlx.categorical_colorbar(categories=ctg, colorscale=colorscale, width=400, height=30, position="bottomleft")

    with urllib.request.urlopen('http://127.0.0.1:8050/static/countries.geojson') as f:
        data = json.load(f)


    for feature in data["features"]:
        feature["properties"][color_col] = covid_data[covid_data["iso_code"] == feature["properties"]["ISO_A3"]][color_col].max()
        feature["properties"]["total_deaths"] = covid_data[covid_data["iso_code"] == feature["properties"]["ISO_A3"]]["total_deaths"].max()


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
                'margin': {'l': 0, 'r': 0, 't': 50, 'b': 0},
            }
    )

    fig.update_layout(title = "Top " + str(k) + " countries oer total cases per million", yaxis = dict(categoryorder = 'total ascending'))
    fig.update_layout(yaxis_visible=False, yaxis_showticklabels=False)

    return [dcc.Graph(figure = fig, id = "left_ranking")]



@app.callback(Output("info", "children"), [Input("geojson", "hover_feature")])
def info_hover(feature):
    return get_info(feature)

@app.callback(Output("rightSide", "children"), [Input("geojson", "click_feature")])
def country_click(feature):
    if feature is not None:
        header = [html.H1(feature["properties"]["ADMIN"])]
        return header + [html.A([html.Img(src = app.get_asset_url("who.png")), html.Br(), html.P("Information from WHO")], target = "_blank", href = "https://www.who.int/countries/" + feature["properties"]["ISO_A3"])]
 

def add_predictions(jsonified_data):

    graph_divs = []
    
    # SARIMAX
    axes = []
    axes.append(Axis("Date", 'data["date"]'))
    axes.append(Axis("Full openness", ['data["Physical_education"]'], ["Full openness"]))

    filters = []
    filters.append(Filter(default_value = ["Norway"], column_name = "location", multi = True))

    graph_infos.append(GraphInfo(dataset = jsonified_data,  title = "SARIMAX", axes = axes, filters = filters))
        
    #graph_divs.append(new_custom_graph())

    graph_divs.append(predict_world_cases("SARIMAX"))


    # Prophet
    axes = []
    axes.append(Axis("Date", 'data["date"]'))
    axes.append(Axis("Full openness", ['data["Physical_education"]'], ["Full openness"]))

    filters = []
    filters.append(Filter(default_value = ["Norway"], column_name = "location", multi = True))

    graph_infos.append(GraphInfo(dataset = jsonified_data,  title = "Prophet", axes = axes, filters = filters))
        
    #graph_divs.append(new_custom_graph())

    graph_divs.append(predict_world_cases("Prophet"))


    '''
    # School open vs stringency
    axes = []
    axes.append(Axis("Date", 'data["date"]'))
    axes.append(Axis("Stringency to physical education availability ratio", ['data["stringency_index"]/data["Physical_education"]'], ["SPE"]))

    filters = []
    filters.append(Filter(default_value = ["Norway"], column_name = "location", multi = True))

    graph_infos.append(GraphInfo(dataset = jsonified_data,  title = "Stringency to physical education availability (SPE) ratio", axes = axes, filters = filters))
        
    graph_divs.append(new_custom_graph())


    # School open vs stringency
    axes = []
    axes.append(Axis("Date", 'data["date"]'))
    axes.append(Axis("Full openness", ['data["Physical_education"]'], ["Full openness"]))

    filters = []
    filters.append(Filter(default_value = ["Norway"], column_name = "location", multi = True))

    graph_infos.append(GraphInfo(dataset = jsonified_data,  title = "Stringency to physical education availability (SPE) ratio", axes = axes, filters = filters))
        
    graph_divs.append(new_custom_graph())



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
    axes.append(Axis("New cases to population ratio", ['data["new_cases"]/data["population"]'], ["New cases by population"]))

    filters = []
    filters.append(Filter(default_value = ["Italy"], column_name = "location", multi = True))

    graph_infos.append(GraphInfo(dataset = jsonified_data,  title = "New cases by population", axes = axes, filters = filters))
        
    graph_divs.append(new_custom_graph())

    
    # Graph 1: new cases to deaths
    axes = []
    axes.append(Axis("Date", 'data["date"]'))
    axes.append(Axis("New deaths to cases ratio [%]", ['100*data["new_deaths"].shift(periods = -11)/data["new_cases"]'], ["New deaths to new cases ratio"], log_scale = True))

    filters = []
    filters.append(Filter(default_value = ["Italy"], column_name = "location", multi = True))

    graph_infos.append(GraphInfo(dataset = jsonified_data,  title = "New deaths to new cases ratio (11 days shifted)", axes = axes, filters = filters))
        
    graph_divs.append(new_custom_graph())


    
    # Graph 1: new increment with respect to previous (hospitalization)
    axes = []
    axes.append(Axis("Date", 'data["date"]'))
    axes.append(Axis("Hospitalized patients increment", ['data["hosp_patients"].diff()/data["hosp_patients"].shift(periods = 1)'], ["New hospitalizations wrt the previous day"]))

    filters = []
    filters.append(Filter(default_value = ["Italy"], column_name = "location", multi = True))

    graph_infos.append(GraphInfo(dataset = jsonified_data,  title = "New hospitalizations wrt the previous day", axes = axes, filters = filters))
        
    graph_divs.append(new_custom_graph())


    
    # Graph 1: new increment with respect to previous (ICUs)
    axes = []
    axes.append(Axis("Date", 'data["date"]'))
    axes.append(Axis("ICU patients increment", ['data["icu_patients"].diff()/data["icu_patients"].shift(periods = 1)'], ["New ICUs wrt the previous day"]))

    filters = []
    filters.append(Filter(default_value = ["Italy"], column_name = "location", multi = True))

    graph_infos.append(GraphInfo(dataset = jsonified_data,  title = "New ICUs wrt the previous day", axes = axes, filters = filters))
        
    graph_divs.append(new_custom_graph())



    # Graph 1: new deaths per million people
    axes = []
    axes.append(Axis("Date", 'data["date"]'))
    axes.append(Axis("Deaths per million", ['data["new_deaths_per_million"]'], ["New deaths"]))

    filters = []
    filters.append(Filter(default_value = ["Norway"], column_name = "location", multi = True))

    graph_infos.append(GraphInfo(dataset = jsonified_data,  title = "Deaths per million", axes = axes, filters = filters))
        
    graph_divs.append(new_custom_graph())
    
    
    # Graph 2: new tests, confirmed cases, deaths per million people
    axes = []
    axes.append(Axis("Date", 'data["date"]'))
    axes.append(Axis("New tests, cases and deaths per million", ['data["new_tests_per_thousand"]*1000', 'data["new_cases_per_million"]', 'data["new_deaths_per_million"]'], ["New tests", "New cases", "New deaths"]))

    filters = []
    filters.append(Filter(default_value = ["Norway"], column_name = "location", multi = True))

    graph_infos.append(GraphInfo(dataset = jsonified_data,  title = "New tests, confirmed cases and deaths per million", axes = axes, filters = filters))
        
    graph_divs.append(new_custom_graph())
    

    # Graph 3: cumulative tests, confirmed cases, deaths per million people
    axes = []
    axes.append(Axis("Date", 'data["date"]'))
    axes.append(Axis("Total tests, cases and deaths per million", ['data["total_tests_per_thousand"]*1000', 'data["total_cases_per_million"]', 'data["total_deaths_per_million"]'], ["Total tests", "Total cases", "Total deaths"]))

    filters = []
    filters.append(Filter(filter_type="DatePickerRange", column_name = "date"))
    filters.append(Filter(default_value = ["Norway"], column_name = "location", multi = True))

    graph_infos.append(GraphInfo(dataset = jsonified_data,  title = "Cumulative tests, confirmed cases and deaths per million", axes = axes, filters = filters))
        
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
    

    # ICU to hospitalizations ratio
    axes = []
    axes.append(Axis("x", 'data["location"]'))
    axes.append(Axis("y", ['data["icu_patients"]/data["hosp_patients"]'], ["Deaths to cases ratio"]))

    filters = []
    graph_infos.append(GraphInfo(dataset = jsonified_data,  title = "ICU to hospitalization ratio", axes = axes, filters = filters, animation_frame = 'data["date"].astype(str)'))
        
    graph_divs.append(new_custom_map())


    
    # ICU to hospitalizations ratio
    axes = []
    axes.append(Axis("x", 'data["location"]'))
    axes.append(Axis("y", ['data["icu_patients"]/data["hosp_patients"]'], ["Deaths to cases ratio"]))

    filters = []
    graph_infos.append(GraphInfo(dataset = jsonified_data,  title = "ICU to hospitalization ratio", axes = axes, filters = filters, animation_frame = 'data["date"].astype(str)'))
        
    graph_divs.append(new_custom_map())
    


    # Population density vs cases per million
    axes = []
    axes.append(Axis("Population density", 'mask_max(data, "total_cases_per_million")["population_density"]'))
    axes.append(Axis("Cases per million", ['[data["total_cases_per_million"].max()]'], ["Cases per million"]))

    filters = []
    filters.append(Filter(show_on_marker = True, default_value = ["Norway", "Italy", "Sweden"], column_name = "location", multi = True))

    graph_infos.append(GraphInfo(dataset = jsonified_data,  title = "Population density vs cases per million", axes = axes, filters = filters, hide_side_legend = True, plot_type = "markers"))
        
    graph_divs.append(new_custom_graph())



    # Stringency and cases per million
    axes = []
    axes.append(Axis("Date", 'data["date"]'))
    axes.append(Axis("Deaths and stringency", ['data["new_deaths_per_million"]', 'data["stringency_index"]'], ["Total deaths", "Stringency index"]))

    filters = []

    filters.append(Filter(default_value = ["Norway"], column_name = "location", multi = True))

    graph_infos.append(GraphInfo(dataset = jsonified_data,  title = "Deaths and stringency index through time", axes = axes, filters = filters))
        
    graph_divs.append(new_custom_graph())

   

    
    # Stringency / deaths (1 week shift)
    axes = []
    axes.append(Axis("Date", 'data["date"]'))
    axes.append(Axis("Stringency per new deaths per million", ['data["stringency_index"]/(1 + data["new_deaths_per_million"].shift(periods = 7))'], ["Stringency index per new deaths"]))

    filters = []
    filters.append(Filter(default_value = ["Italy"], column_name = "location", multi = True))

    graph_infos.append(GraphInfo(dataset = jsonified_data,  title = "Stringency per new deaths (1 week shift)", axes = axes, filters = filters))
        
    graph_divs.append(new_custom_graph())


    # Stringency / deaths (no shift)
    axes = []
    axes.append(Axis("Date", 'data["date"]'))
    axes.append(Axis("Stringency per new deaths per million", ['data["stringency_index"]/(1 + data["new_deaths_per_million"])'], ["Stringency index per new deaths"]))

    filters = []
    filters.append(Filter(default_value = ["Italy"], column_name = "location", multi = True))

    graph_infos.append(GraphInfo(dataset = jsonified_data,  title = "Stringency per new deaths (no shift)", axes = axes, filters = filters))
        
    graph_divs.append(new_custom_graph())



    '''

    # Initialize preset container and return
    preset_container = html.Div(children = graph_divs)
    return [preset_container]
    
def add_analyses(jsonified_data):
    graph_divs = []
    # Graph 1: new increment with respect to previous (hospitalization)
    axes = []
    axes.append(Axis("Date", 'data["date"]'))
    axes.append(Axis("Hospitalized patients increment", ['data["hosp_patients"].diff()/data["hosp_patients"].shift(periods = 1)'], ["New hospitalizations wrt the previous day"]))

    filters = []
    filters.append(Filter(default_value = ["Italy"], column_name = "location", multi = True))

    graph_infos.append(GraphInfo(dataset = jsonified_data,  title = "New hospitalizations wrt the previous day", axes = axes, filters = filters))
        
    graph_divs.append(new_custom_graph())


    # Graph 3: cumulative tests, confirmed cases, deaths per million people
    axes = []
    axes.append(Axis("Date", 'data["date"]'))
    axes.append(Axis("Tests, cases and deaths per million", ['data["total_tests_per_thousand"]*1000', 'data["total_cases_per_million"]', 'data["total_deaths_per_million"]'], ["Total tests", "Total cases", "Total deaths"], log_scale = True))

    filters = []
    filters.append(Filter(filter_name = "Median age", filter_type="RangeSlider", column_name = "median_age"))

    filters.append(Filter(default_value = ["Norway"], column_name = "location", multi = True))

    graph_infos.append(GraphInfo(dataset = jsonified_data,  title = "Cumulative tests, confirmed cases and deaths per million", axes = axes, filters = filters))
        
    graph_divs.append(new_custom_graph())


    # Initialize preset container and return
    preset_container = html.Div(children = graph_divs)
    return [preset_container]



def add_education(jsonified_data):
    graph_divs = []

    # School open vs stringency
    axes = []
    axes.append(Axis("Date", 'data["date"]'))
    axes.append(Axis("Stringency to physical education availability ratio", ['data["stringency_index"]/data["Physical_education"]'], ["SPE"]))

    filters = []
    filters.append(Filter(default_value = ["Norway"], column_name = "location", multi = True))

    graph_infos.append(GraphInfo(dataset = jsonified_data,  title = "Stringency to physical education availability (SPE) ratio", axes = axes, filters = filters))
        
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
    min_color = max(graph_info.min_animation, color_data.mean() - 2*color_data.std())
    max_color = min(graph_info.max_animation, color_data.mean() + 2*color_data.std())
   
    fig = px.choropleth(data, range_color = [min_color, max_color], animation_frame = eval(graph_info.animation_frame), locations = eval(graph_info.axes[0].content).values, color = color_data.values, color_continuous_scale = graph_info.colorscale, locationmode = graph_info.location_mode)
    
    fig.update_layout(margin={"r":5,"t":60,"l":5,"b":5}, title = graph_info.title)

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
            filter_menu = dcc.RangeSlider(tooltip = {'always_visible': True, 'placement': 'bottom'}, dots = True, id =  {'type': 'SR', 'index': ind, 'internal_index': subindex}, min=floor(data[f.column_name].min()*0.9), max=ceil(data[f.column_name].max()*1.1), step=1, value=[floor(data[f.column_name].min()), ceil(data[f.column_name].max())])
        
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


            for defv in def_val_list:
                data = covid_data[covid_data[f.column_name] == defv]

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
                        
                fig.add_trace(go.Scatter(mode = graph_info.plot_type, name=label, x=eval(graph_info.axes[0].content), y=eval(y_trace)))
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


def predict_world_cases(prediction_method):

    print(prediction_method)
    ind = len(graph_infos)-1
    graph_info = graph_infos[ind]
    covid_data = pd.read_json(graph_info.dataset, orient="split")


    #graph_div = dash_draggable.dash_draggable(axis="both", grid=[30, 30], children = [])
    graph_div = html.Div(children = [])
    #fig = px.line(title = graph_info.title)
    fig = go.Figure(layout = {'title': "Prediction of the world cases in the next 6 months"})

    dataset = covid_data.loc[covid_data['date'] >= pd.to_datetime(datetime.date(2020, 1, 22))]
    dataset = dataset.loc[covid_data['location'] == "Italy"]

    dataset[["new_cases", "stringency_index"]].fillna(0, inplace = True)
    dataset["total_cases"].fillna(method = 'bfill', inplace = True)


    # Only dates and values (simple case but no good results)
    prediction_dataset = pd.DataFrame(columns = ['date', 'value'])
    dates = list(dataset['date'])
    prediction_dataset['date'] = dates
    prediction_dataset['value'] = dataset["new_cases"].to_list()
    prediction_dataset.set_index("date", inplace = True)

    # Multiple attributes
    pred_multi_dataset = pd.DataFrame(columns = ['date', 'value', 'stringency'])
    dates = list(dataset['date'])
    pred_multi_dataset['date'] = dates
    pred_multi_dataset['value'] = dataset["new_cases"].to_list()
    pred_multi_dataset['stringency'] = dataset["stringency_index"].to_list()
    pred_multi_dataset.set_index("date", inplace = True)



    start_date = "2020-11-03"

    train = pred_multi_dataset.loc[pred_multi_dataset.index < pd.to_datetime(start_date)]
    test = pred_multi_dataset.loc[pred_multi_dataset.index >= pd.to_datetime(start_date)]

    train.fillna(0, inplace = True)
    test.fillna(0, inplace = True)

    if(prediction_method == "SARIMAX"):

        train.drop(columns = ["stringency"], inplace = True)
        test.drop(columns = ["stringency"], inplace = True)
        #model = pm.auto_arima(train, start_p = 1, start_q = 1, test = 'adf', max_p = 10, max_q = 10, m=1, d=None, seasonal=False, start_P = 0, D=0, trace = True, error_action = 'ignore', suppress_warning=True, stepwise = True)
        #print(model.summary())

        model = SARIMAX(train, order=(2, 2, 10)) 
    
        # New cases per day
        # World (7, 1, 8), Italy (1, 2, 0), Norway (2, 2, 3)

        results = model.fit(disp = False)

        sarimax_prediction = results.predict(start = start_date, end='2021-06-01', dynamic=False)
        fig.add_trace(go.Scatter(mode = graph_info.plot_type, name="Prediction", x=sarimax_prediction.index, y=sarimax_prediction))

    elif(prediction_method == "Prophet"):
        train["ds"] = train.index
        train.rename(columns = {'date': 'ds', 'value': 'y'}, inplace = True)
        test.rename(columns = {'date': 'ds', 'value': 'y'}, inplace = True)

        print(train.columns)
        m = Prophet()
        # https://stackoverflow.com/questions/54544285/is-it-possible-to-do-multivariate-multi-step-forecasting-using-fb-prophet to add parameters

        m.fit(train)
        future = m.make_future_dataframe(periods = len(test) - 1)
        forecast = m.predict(future)
        fig.add_trace(go.Scatter(mode = graph_info.plot_type, name="Prediction", x=forecast["ds"], y=forecast["yhat"]))

    elif(prediction_method == "Linear"):
        # Look at the past and predict + LSTM
        train_data_all = dataset.loc[dataset["date"] < pd.to_datetime(start_date)]
        test_data_all = dataset.loc[dataset["date"] >= pd.to_datetime(start_date)]
        
        model = sk.linear_model.LinearRegression()

        y = train_data_all["new_cases"].values.reshape(len(train_data_all), 1)
        x = train_data_all[["stringency_index", "total_cases"]].values.reshape(len(train_data_all), 2)

        model.fit(x, y)

        intercept = model.intercept_
        coeff = model.coef_

        rss = multivarRSS(test_data_all["new_cases"], test_data_all[["stringency_index", "total_cases"]], intercept, coeff)
        test_data_all["predicted"] = intercept[0] + test_data_all["total_cases"]*coeff[0][1] + test_data_all["stringency_index"]*coeff[0][0]

        print(test_data_all)
        fig.add_trace(go.Scatter(mode = graph_info.plot_type, name="Prediction", x=test_data_all["date"], y=test_data_all["predicted"]))



    fig.add_trace(go.Scatter(mode = graph_info.plot_type, name="Observation", x=dataset["date"], y=dataset["new_cases"]))

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


    Input("saved_data", "children")]
    )
    
def update_graph(filter_value, filter_id, start_date, end_date, date_id, slider_id, slider_value, jsonified_data):
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

                        rgb_color = plotly.colors.DEFAULT_PLOTLY_COLORS[j%10].lstrip("rgb(").rstrip(")").split(", ")
                        rgb_color = list(map(int, rgb_color))

                        computed_color = "rgb(" + str(rgb_color[0]*(1+lab/4)) + ", " + str(rgb_color[1]*(1+lab/4)) + ", " + str(rgb_color[2]*(1+lab/4)) + ")"


                        color_properties = graph_info.plot_type[:-1] + "=dict(color = '" + computed_color + "')"
                        fig.add_trace(go.Scatter(line = dict(color = computed_color), mode=graph_info.plot_type, name=label, x=eval(graph_info.axes[0].content), y=eval(y_trace)))

         

        fig.update_layout(showlegend=not graph_info.hide_side_legend)
        if(graph_info.axes[0].log_scale is True):
            fig.update_xaxes(type="log")

        if(graph_info.axes[1].log_scale is True):
            fig.update_yaxes(type="log")
        
        return fig
    else:
        raise dash.exceptions.PreventUpdate




    



if __name__ == '__main__':
    app.run_server(debug=True)