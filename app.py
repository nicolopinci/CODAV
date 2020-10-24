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
            'Drag and Drop or ',
            html.A('Select Files')
        ]),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
        # Do not allow multiple files to be uploaded
        multiple=False
        ),

        html.Div(id='output-data-upload'),
        html.H1(children=project_name),
        #html.Button("Add graph", id='load-new-content', n_clicks=0)

    ],
)



def parse_contents(contents):
    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)

    try:
        covid_data = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        col_options = [dict(label=x, value=x) for x in covid_data.columns]
        app.layout = new_scatter(app.layout, covid_data, "location", "total_cases", "new_deaths")

    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])




@app.callback(Output('output-data-upload', 'children'),
              [Input('upload-data', 'contents')])

def update_output(content):

    if content is not None:
        parse_contents(content)


'''
@app.callback(
    Output('top_container','children'),
    [Input('load-new-content','n_clicks')],
    [State('top_container','children')])
'''

def new_scatter(old_output, covid_data, x_col, y_col, color_data=None):
   
    fig = px.scatter(covid_data, x=x_col, y=y_col, color=color_data)


    graph = dcc.Graph(figure=fig)
    graph.className = "graph"

    move_button = html.I("")
    move_button.className = "moveGraph fas fa-arrows-alt"

    graph_div = html.Div(className="graph_div", children=[move_button])
    graph_div.children.append(graph)

    old_output.children.append(graph_div)

    return old_output





if __name__ == '__main__':
    app.run_server(debug=True)