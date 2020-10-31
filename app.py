# -*- coding: utf-8 -*-

# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

import base64
import datetime
import io

import dash
from dash.dependencies import Input, Output, State, MATCH

import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
import pandas as pd
import dash_table
import dash_daq as daq
import dash_draggable

import plotly.graph_objs as go
from dash.exceptions import PreventUpdate

covid_data = None
dimensions = []

external_stylesheets = ["static/style.css", "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.14.0/css/all.min.css"]
external_scripts = ["static/moveGraphs.js"]

app = dash.Dash(__name__, external_stylesheets=external_stylesheets, external_scripts=external_scripts)


# Serving local static files
@app.server.route('/static/<path:path>')
def static_file(path):
    static_folder = os.path.join(os.getcwd(), 'static')
    return send_from_directory(static_folder, path)


# see https://plotly.com/python/px-arguments/ for more options

project_name = "CODAV"

app.layout = html.Div(
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



def parse_contents(contents):
    print("parsing contents")
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
    ind = 0
    graph_divs.append(new_population_per_country(jsonified_data, "location", "date", "new_cases_per_million", ind))
    ind = 1
    graph_divs.append(new_population_per_country(jsonified_data, "location", "date", "new_deaths_per_million", ind))
    
    return graph_divs

def new_population_per_country(jsonified_data, column_filter, x_col, y_col, ind):
    
    covid_data = pd.read_json(jsonified_data, orient="split")

    dropdown_menu = dcc.Dropdown(id={'type': 'DD', 'index': ind}, value=['Norway'], options = [{'label': i, 'value': i} for i in covid_data["location"].unique()], multi=True)

    filtered_data = covid_data[covid_data[column_filter] == dropdown_menu.value[0]]
    
    fig = px.line(filtered_data, x=x_col, y=y_col)

    select_quantity = dcc.RadioItems(options=[
        {'label': 'Confirmed cases', 'value': 'cases'},
        {'label': 'Confirmed deaths', 'value': 'deaths'}
        ],
        value='deaths',
       id={'type': 'RD', 'index': ind}
    )

    graph = dcc.Graph(id={'type': 'GR', 'index': ind})
    graph.className = "graph_div graph"

    resize_button = html.I("")
    resize_button.className = "resizeGraph fas fa-expand-alt"

    move_button = html.I("")
    move_button.className = "moveGraph fas fa-arrows-alt"


    graph_div = dash_draggable.dash_draggable(axis="both", grid=[30, 30], children=[move_button, resize_button])
    
    graph_div.children.append(dropdown_menu)
    graph_div.children.append(graph)
    graph_div.children.append(select_quantity)

    return graph_div



@app.callback( 
Output({'type': 'GR', 'index': MATCH}, component_property='figure'), 
    [Input({'type':'DD', 'index': MATCH}, component_property='value'),
    Input({'type':'RD',  'index': MATCH}, component_property='value'),
    Input("saved_data", "children")]) 
def update_graph(filter_value, radio_value, jsonified_data):
    covid_data = pd.read_json(jsonified_data, orient="split")

    fig = px.line(title = "Number of " + radio_value + " per million people through time")
    fig.update_layout(xaxis_title='Date', yaxis_title="New " + radio_value + " per million people")

    if(covid_data is not None):

        for i in range(0, len(filter_value)):
            filtered_data = covid_data[covid_data["location"] == filter_value[i]]
            fig.add_trace(go.Scatter(x=filtered_data["date"], y=filtered_data["new_" + radio_value + "_per_million"], name=filter_value[i], showlegend=True))
        
        return fig
    else:
        raise dash.exceptions.PreventUpdate




if __name__ == '__main__':
    app.run_server(debug=True)