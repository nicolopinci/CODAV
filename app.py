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
import plotly.express as px
import pandas as pd
import dash_table
import dash_daq as daq
import dash_draggable
import json
import pickle

import plotly.graph_objs as go
from dash.exceptions import PreventUpdate

covid_data = None
dimensions = []
graph_infos = []


external_stylesheets = ["static/style.css", "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.14.0/css/all.min.css"]
external_scripts = ["static/moveGraphs.js"]

app = dash.Dash(__name__, external_stylesheets=external_stylesheets, external_scripts=external_scripts)



class GraphInfo:
    def __init__(self, dataset, title, graph_type = "line", axes = [], color = None, filters = [], animation = None, map_type = 'choropleth', location_mode = 'country names', colorscale='Portland', animation_frame = 'data["date"].astype(str)', min_animation = 0, max_animation = 1):
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
    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True, indent=4)

class Axis:
    def __init__(self, label = "", log_scale = False, content = [], labels = []):
        self.label = label
        self.log_scale = log_scale
        self.content = content
        self.labels = labels

class Filter:
    def __init__(self, column_name = "", default_value = [], multi = True, filter_type="Dropdown", start_date = "2020-01-01", end_date="2020-12-31"):
        self.column_name = column_name
        self.default_value = default_value
        self.multi = multi
        self.filter_type = filter_type
        self.start_date = start_date
        self.end_date = end_date

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
    in_data[in_data[filter_key] != in_data[filter_key]] = in_data[filter_key].min()
    in_data[in_data[filter_key] != in_data[filter_key]] = 0

    out_data = in_data[in_data[filter_key] == in_data[filter_key].max()]
    return out_data

def generate_layout():
    return html.Div(
    id="top_container",
    children=[

	    dcc.Upload(
        id='upload-data',
        children=html.Div([
            html.I("", className="fas fa-upload")
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

        html.Div(id="graphs_container", children=[]),
        html.Div(id="filter_equivalence", style={'display': 'none'}),

        # Hidden div inside the app that stores the intermediate value
        html.Div(id='saved_data', style={'display': 'none'})
    ],
    )

app.layout = generate_layout


def parse_contents(contents):
    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)

    try:
        df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
      
        return df.to_json(date_format='iso', orient='split')


    except Exception as e:
        return html.Div([
            'There was an error processing this file.'
        ])




@app.callback(Output('saved_data', 'children'),
              [Input('upload-data', 'contents')])
def update_output(contents):
    if contents is not None:
        return parse_contents(contents)
    else:
        raise dash.exceptions.PreventUpdate




    
@app.callback([Output("graphs_container", "children")],
[Input("saved_data", "children")])
def add_preset(jsonified_data):

    graph_divs = []
 
    # Graph 3: cumulative tests, confirmed cases, deaths per million people
    axes = []
    axes.append(Axis("x", True, 'data["date"]'))
    axes.append(Axis("y", True, ['data["total_tests_per_thousand"]*1000', 'data["total_cases_per_million"]', 'data["total_deaths_per_million"]'], ["Total tests", "Total cases", "Total deaths"]))

    filters = []
    #filters.append(Filter(filter_type="DatePickerRange", column_name = "date"))
    filters.append(Filter(filter_type="RangeSlider", column_name = "median_age"))

    filters.append(Filter(default_value = ["Norway"], column_name = "location", multi = True))

    graph_infos.append(GraphInfo(dataset = jsonified_data, title = "Cumulative tests, confirmed cases and deaths per million", axes = axes, filters = filters))
        
    graph_divs.append(new_custom_graph())

    # Graph 1: new cases by population
    axes = []
    axes.append(Axis("x", True, 'data["date"]'))
    axes.append(Axis("y", True, ['data["new_cases"]/data["population"]'], ["New cases by population"]))

    filters = []
    filters.append(Filter(default_value = ["Italy"], column_name = "location", multi = True))

    graph_infos.append(GraphInfo(dataset = jsonified_data, title = "New cases by population", axes = axes, filters = filters))
        
    graph_divs.append(new_custom_graph())


    # Graph 1: new increment with respect to previous (hospitalization)
    axes = []
    axes.append(Axis("x", True, 'data["date"]'))
    axes.append(Axis("y", True, ['data["hosp_patients"].diff()/data["hosp_patients"].shift(periods = 1)'], ["New hospitalizations wrt the previous day"]))

    filters = []
    filters.append(Filter(default_value = ["Italy"], column_name = "location", multi = True))

    graph_infos.append(GraphInfo(dataset = jsonified_data, title = "New hospitalizations wrt the previous day", axes = axes, filters = filters))
        
    graph_divs.append(new_custom_graph())


    
    # Graph 1: new increment with respect to previous (ICUs)
    axes = []
    axes.append(Axis("x", True, 'data["date"]'))
    axes.append(Axis("y", True, ['data["icu_patients"].diff()/data["icu_patients"].shift(periods = 1)'], ["New ICUs wrt the previous day"]))

    filters = []
    filters.append(Filter(default_value = ["Italy"], column_name = "location", multi = True))

    graph_infos.append(GraphInfo(dataset = jsonified_data, title = "New ICUs wrt the previous day", axes = axes, filters = filters))
        
    graph_divs.append(new_custom_graph())



    # Graph 1: new deaths per million people
    axes = []
    axes.append(Axis("x", True, 'data["date"]'))
    axes.append(Axis("y", True, ['data["new_deaths_per_million"]'], ["New deaths"]))

    filters = []
    filters.append(Filter(default_value = ["Norway"], column_name = "location", multi = True))

    graph_infos.append(GraphInfo(dataset = jsonified_data, title = "Deaths per million", axes = axes, filters = filters))
        
    graph_divs.append(new_custom_graph())
    
    
    # Graph 2: new tests, confirmed cases, deaths per million people
    axes = []
    axes.append(Axis("x", True, 'data["date"]'))
    axes.append(Axis("y", True, ['data["new_tests_per_thousand"]*1000', 'data["new_cases_per_million"]', 'data["new_deaths_per_million"]'], ["New tests", "New cases", "New deaths"]))

    filters = []
    filters.append(Filter(default_value = ["Norway"], column_name = "location", multi = True))

    graph_infos.append(GraphInfo(dataset = jsonified_data, title = "New tests, confirmed cases and deaths per million", axes = axes, filters = filters))
        
    graph_divs.append(new_custom_graph())
    

    # Graph 3: cumulative tests, confirmed cases, deaths per million people
    axes = []
    axes.append(Axis("x", True, 'data["date"]'))
    axes.append(Axis("y", True, ['data["total_tests_per_thousand"]*1000', 'data["total_cases_per_million"]', 'data["total_deaths_per_million"]'], ["Total tests", "Total cases", "Total deaths"]))

    filters = []
    filters.append(Filter(filter_type="DatePickerRange", column_name = "date"))
    filters.append(Filter(default_value = ["Norway"], column_name = "location", multi = True))

    graph_infos.append(GraphInfo(dataset = jsonified_data, title = "Cumulative tests, confirmed cases and deaths per million", axes = axes, filters = filters))
        
    graph_divs.append(new_custom_graph())
    
    
    # 4: Map with deaths per million
    axes = []
    axes.append(Axis("x", True, 'data["location"]'))
    axes.append(Axis("y", True, ['data["total_deaths_per_million"]'], ["Total deaths per million"]))

    filters = []
    #filters.append(Filter(default_value = ["2020-10-19"], column_name = "date", multi = True))

    graph_infos.append(GraphInfo(dataset = jsonified_data, title = "Total deaths per million", axes = axes, filters = filters, animation_frame = 'data["date"].astype(str)'))
        
    graph_divs.append(new_custom_map())


    # 5: Map with cases per million
    axes = []
    axes.append(Axis("x", True, 'data["location"]'))
    axes.append(Axis("y", True, ['data["total_cases_per_million"]'], ["Total cases per million"]))

    filters = []

    graph_infos.append(GraphInfo(dataset = jsonified_data, title = "Total cases per million", axes = axes, filters = filters))
        
    graph_divs.append(new_custom_map())



    # 5: Deaths to cases ratio
    axes = []
    axes.append(Axis("x", True, 'data["location"]'))
    axes.append(Axis("y", True, ['data["total_deaths"]/data["total_cases"]'], ["Deaths to cases ratio"]))

    filters = []
    graph_infos.append(GraphInfo(dataset = jsonified_data, title = "Deaths to cases ratio", axes = axes, filters = filters, animation_frame = 'data["date"].astype(str)'))
        
    graph_divs.append(new_custom_map())

    # ICU to hospitalizations ratio
    axes = []
    axes.append(Axis("x", True, 'data["location"]'))
    axes.append(Axis("y", True, ['data["icu_patients"]/data["hosp_patients"]'], ["Deaths to cases ratio"]))

    filters = []
    graph_infos.append(GraphInfo(dataset = jsonified_data, title = "ICU to hospitalization ratio", axes = axes, filters = filters, animation_frame = 'data["date"].astype(str)'))
        
    graph_divs.append(new_custom_map())


    
    # ICU to hospitalizations ratio
    axes = []
    axes.append(Axis("x", True, 'data["location"]'))
    axes.append(Axis("y", True, ['data["icu_patients"]/data["hosp_patients"]'], ["Deaths to cases ratio"]))

    filters = []
    graph_infos.append(GraphInfo(dataset = jsonified_data, title = "ICU to hospitalization ratio", axes = axes, filters = filters, animation_frame = 'data["date"].astype(str)'))
        
    graph_divs.append(new_custom_map())



    # Population density vs cases per million
    axes = []
    axes.append(Axis("x", True, '[mask_max(data, "total_cases_per_million")["population_density"]]'))
    axes.append(Axis("y", True, ['[data["total_cases_per_million"].max()]'], ["Cases per million"]))

    filters = []
    filters.append(Filter(default_value = 'data["location"].unique()', column_name = "location", multi = True))

    graph_infos.append(GraphInfo(dataset = jsonified_data, title = "Population density vs cases per million", axes = axes, filters = filters))
        
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

            filter_menu = dcc.RangeSlider(id =  {'type': 'SR', 'index': ind, 'internal_index': subindex}, min=data[f.column_name].min()*0.9, max=data[f.column_name].max()*1.1, step=(data[f.column_name].diff().max() - data[f.column_name].diff().min())/1000.0, value=[data[f.column_name].min(), data[f.column_name].max()])
        
        graph_div.children.append(filter_menu)

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
                    label += ", " + defv

          

                fig.add_trace(go.Scatter(mode = 'markers', name=label, x=eval(graph_info.axes[0].content), y=eval(y_trace)))
        subindex += 1

        covid_data = data
        
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
    fig = go.Figure(layout = {'title': graph_info.title})


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
                            label += ", " + filter_value[i][j]
                    
                        fig.add_trace(go.Scatter(mode='markers', name=label, x=eval(graph_info.axes[0].content), y=eval(y_trace)))

         


        return fig
    else:
        raise dash.exceptions.PreventUpdate




    



if __name__ == '__main__':
    app.run_server(debug=True)