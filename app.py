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

import plotly.graph_objs as go
from dash.exceptions import PreventUpdate

covid_data = None
dimensions = []

external_stylesheets = ["static/style.css", "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.14.0/css/all.min.css"]
external_scripts = ["static/moveGraphs.js"]

app = dash.Dash(__name__, external_stylesheets=external_stylesheets, external_scripts=external_scripts)

class GraphInfo:
    def __init__(self, dataset, title, graph_type = "line", axes = [], color = None, filters = [], animation = None):
        self.title = title 
        self.axes = axes
        self.color = color
        self.filters = filters
        self.animation = animation
        self.graph_type = graph_type
        self.dataset = dataset
    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True, indent=4)

class Axis:
    def __init__(self, label = "", log_scale = False, content = ""):
        self.label = label
        self.log_scale = log_scale
        self.content = content

class Filter:
    def __init__(self, column_name = "", default_value = None, multi = True):
        self.column_name = column_name
        self.default_value = default_value
        self.multi = multi

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
        print(e)
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




    
@app.callback(Output("graphs_container", "children"), [Input("saved_data", "children")])
def add_preset(jsonified_data):

    graph_divs = []
    '''
    ind = 0
    graph_divs.append(new_population_per_country(jsonified_data, "location", "date", "new_cases_per_million", ind))
    ind = 1
    graph_divs.append(new_population_per_country(jsonified_data, "location", "date", "new_deaths_per_million", ind))
    '''

    axes = []
    axes.append(Axis("x", True, 'data[data == "new_deaths_per_million"]'))
    axes.append(Axis("y", True, 'data[data == "location"]'))

    filters = []
    filters.append(Filter(default_value = ["Norway"], column_name = "location", multi = True))
    filters.append(Filter(default_value = ["Canada"], column_name = "location", multi = True))

    gi1 = GraphInfo(dataset = jsonified_data, title = "Deaths per million", axes = axes, filters = filters)


    graph_divs.append(new_custom_graph(gi1, 0))
    return graph_divs
    

def new_custom_graph(graph_info, ind):
    covid_data = pd.read_json(graph_info.dataset, orient="split")

    graph_div = dash_draggable.dash_draggable(axis="both", grid=[30, 30], children = [])
    fig = px.line(title = graph_info.title)

    subindex = 0
    for f in graph_info.filters:
        dropdown_menu = dcc.Dropdown(id={'type': 'DD', 'index': ind, 'internal_index': subindex}, value=f.default_value, options = [{'label': i, 'value': i} for i in covid_data[f.column_name].unique()], multi=f.multi)
        graph_div.children.append(dropdown_menu)
        data = covid_data

        for defv in f.default_value:
            data = covid_data[covid_data[f.column_name] == defv]

        fig.add_trace(go.Scatter(x=eval(graph_info.axes[0].content), y=eval(graph_info.axes[1].content)))
        subindex += 1

    graph = deg.ExtendableGraph(id={'type': 'GR', 'index': ind})
    graph.className = "graph_div graph"

    resize_button = html.I("")
    resize_button.className = "resizeGraph fas fa-expand-alt"

    move_button = html.I("")
    move_button.className = "moveGraph fas fa-arrows-alt"

    graph_div.children.append(move_button)
    graph_div.children.append(resize_button)
    graph_div.children.append(graph)

    return graph_div



@app.callback( 
Output({'type': 'GR', 'index': MATCH}, 'extendData'), 
    [Input({'type':'DD', 'index': MATCH, 'internal_index': ALL}, 'value'),
    Input("saved_data", "children")],
    [State({'type':'GR', 'index': MATCH}, 'figure')]) 

def update_graph(filter_value, jsonified_data, figure):
    covid_data = pd.read_json(jsonified_data, orient="split")

    fig = figure
    print(figure)

    if(covid_data is not None):
        for i in range(0, len(filter_value)):
            for j in range(0, len(filter_value[i])):
                filtered_data = covid_data[covid_data["location"] == filter_value[i][j]]
                fig.add_trace(go.Scatter(x=filtered_data["date"], y=filtered_data["new_" + "deaths" + "_per_million"], name=filter_value[i][j], showlegend=True))
        
        return fig
    else:
        raise dash.exceptions.PreventUpdate




if __name__ == '__main__':
    app.run_server(debug=True)