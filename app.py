# -*- coding: utf-8 -*-

# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

import base64
import datetime
import io

import dash
from dash.dependencies import Input, Output, State

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
        
        #col_options = [dict(label=x, value=x) for x in df.columns]
        #app.layout = new_scatter(app.layout, covid_data, "location", "total_cases", "new_deaths")
        #app.layout = create_filtered(app.layout, covid_data)
        #return daily_new_per_population_country(covid_data)

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



def new_scatter(old_output, covid_data, x_col, y_col, color_data=None):
   
    fig = px.scatter(covid_data, x=x_col, y=y_col, color=color_data)

    graph = dcc.Graph(figure=fig)
    graph.className = "graph_div graph"

    resize_button = html.I("")
    resize_button.className = "resizeGraph fas fa-expand-alt"

    move_button = html.I("")
    move_button.className = "moveGraph fas fa-arrows-alt"

    graph_div = dash_draggable.dash_draggable(axis="both", grid=[30, 30], children=[move_button, resize_button])
    graph_div.children.append(graph)

    old_output.children.append(graph_div)

    return old_output




def create_filtered(old_output, covid_data):
    
    dropdown_menu = dcc.Dropdown(value='Norway', options = [{'label': i, 'value': i} for i in covid_data["location"].unique()], multi=False)

    print(dropdown_menu)
    #dropdown = covid_data[covid_data['location'] == date_picker_value]
    filtered_data = covid_data[covid_data["location"] == dropdown_menu.value]
    fig = px.scatter(filtered_data, x="date", y="new_cases")

    graph = dcc.Graph(figure=fig)
    graph.className = "graph_div graph"

    resize_button = html.I("")
    resize_button.className = "resizeGraph fas fa-expand-alt"

    move_button = html.I("")
    move_button.className = "moveGraph fas fa-arrows-alt"


    graph_div = dash_draggable.dash_draggable(axis="both", grid=[30, 30], children=[move_button, resize_button])
    graph_div.children.append(dropdown_menu)
    graph_div.children.append(graph)

    old_output.children.append(graph_div)

    return old_output
    
@app.callback(Output("graphs_container", "children"), [Input("saved_data", "children")])
def daily_new_per_population_country(jsonified_data):

    covid_data = pd.read_json(jsonified_data, orient="split")

    identifier = "daily_new_population_per_country"
    dropdown_menu = dcc.Dropdown(id='DD_' + identifier, value='Norway', options = [{'label': i, 'value': i} for i in covid_data["location"].unique()], multi=False)

    #dropdown = covid_data[covid_data['location'] == date_picker_value]
    filtered_data = covid_data[covid_data["location"] == dropdown_menu.value]
    
    fig = px.line(filtered_data, x="date", y="new_cases_per_million")

    graph = dcc.Graph(id="GR_" + identifier, figure=fig)
    graph.className = "graph_div graph"

    resize_button = html.I("")
    resize_button.className = "resizeGraph fas fa-expand-alt"

    move_button = html.I("")
    move_button.className = "moveGraph fas fa-arrows-alt"


    graph_div = dash_draggable.dash_draggable(axis="both", grid=[30, 30], children=[move_button, resize_button])
    graph_div.children.append(dropdown_menu)
    graph_div.children.append(graph)

    return graph_div

@app.callback( 
Output(component_id='GR_daily_new_population_per_country', component_property='figure'), 
    [Input(component_id='DD_daily_new_population_per_country', component_property='value')]) 
def update_graph(filter_value):
    print(filter_value)
    if(covid_data is not None):
        filtered_data = covid_data[covid_data["location"] == filter_value]
        return px.line(filtered_data, x="date", y="new_cases_per_million")
    else:
        raise dash.exceptions.PreventUpdate




if __name__ == '__main__':
    app.run_server(debug=True)