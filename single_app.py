import os
import random

import base64
import dash
from dash import Dash, dcc, html, Input, Output, no_update
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import pandas as pd

from src.utils import cpath, load_pkl

url_default_functions={}
analysis_name = 'bhairaviTransposed'
clus_type = 'kmeans'

app = Dash(__name__)

data_dict = {'df': None, 'ap_series': None, 'pp_series': None, 'clusters': None}

svara_header = html.Div([html.H4(id="data-text", children="Svara", style={'font-family':'sans-serif'})], style={'display':'flex', 'height':45})
svara_dd = dcc.Dropdown(
           id = 'input-data',
           value='All',
           placeholder='Select svara type',
           options= [
               {'label' : 'Sa', 'value' : 'Sa'},
               {'label' : 'Ri', 'value' : 'Ri'},
               {'label' : 'Ga', 'value' : 'Ga'},
               {'label' : 'Ma', 'value' : 'Ma'},
               {'label' : 'Pa', 'value' : 'Pa'},
               {'label' : 'Dha', 'value' : 'Dha'},
               {'label' : 'Ni', 'value' : 'Ni'},
               {'label' : 'All', 'value' : 'All'}], style={'width':500, 'font-family':'sans-serif', "margin-bottom": 200})

tooltip = dcc.Tooltip(id="graph-tooltip")
placeholder = html.Div(id="placeholder", style={"display": "none"})
graph = dbc.Card(dcc.Graph(id="scatter-graph", clear_on_unhover=True))

app.layout = dbc.Container(
[
    dbc.Row([dbc.Col([svara_header, svara_dd, graph])], style={'display':'flex'}),
    tooltip,
    placeholder
])


@app.callback(
    Output("graph-tooltip", "show"),
    Output("graph-tooltip", "bbox"),
    Output("graph-tooltip", "children"),
    Input("scatter-graph", "hoverData"),
)
def display_hover(hoverData):
    if hoverData is None:
        return False, no_update, no_update
    
    # demo only shows the first point, but other points may also be available
    pt = hoverData["points"][0]
    bbox = pt["bbox"]
    num = pt["pointNumber"]

    df_row = data_dict['df'].iloc[num]
    cn = pt['curveNumber']
    pi = pt['pointIndex']

    img_src = os.path.join(os.getcwd(), data_dict['pp_series'][cn][pi])
    
    encoded_image = base64.b64encode(open(img_src, 'rb').read())

    children = [
        html.Div([
            html.Img(src='data:image/png;base64,{}'.format(encoded_image.decode()), style={"width": "100%"}),
        ], style={'width': '800px', 'whiteSpace': 'normal'})
    ]

    return True, bbox, children


@app.callback(
    Output("placeholder", "children"),
    Input("scatter-graph", "clickData"),
)
def fig_click(clickData):
    if not clickData:
        raise dash.exceptions.PreventUpdate
    
    pt = clickData["points"][0]
    cn = pt['curveNumber']
    pi = pt['pointIndex']
    audio_src = os.path.join(os.getcwd(), data_dict['ap_series'][cn][pi])

    encoded_sound = base64.b64encode(open(audio_src, 'rb').read())
    return html.Audio(src='data:audio/mpeg;base64,{}'.format(encoded_sound.decode()),
                          controls=False,
                          autoPlay=True,
                          )

@app.callback(
    Output('scatter-graph', 'figure'),
    [Input('input-data', 'value')]
)
def choose_plot_data(input_data):
    
    disab = False

    if input_data is None:
        raise dash.exceptions.PreventUpdate()
    if input_data == 'Sa':
        data_path = f'data/analysis/{analysis_name}/clusters_single/{clus_type}/embedding_kmeans_sa.csv'
        this_direc = f'data/analysis/{analysis_name}/clusters_single/{clus_type}/sa/'
    if input_data == 'Ri':
        data_path = f'data/analysis/{analysis_name}/clusters_single/{clus_type}/embedding_kmeans_ri.csv'
        this_direc = f'data/analysis/{analysis_name}/clusters_single/{clus_type}/ri/'
    if input_data == 'Ga':
        data_path = f'data/analysis/{analysis_name}/clusters_single/{clus_type}/embedding_kmeans_ga.csv'
        this_direc = f'data/analysis/{analysis_name}/clusters_single/{clus_type}/ga/'
    if input_data == 'Ma':
        data_path = f'data/analysis/{analysis_name}/clusters_single/{clus_type}/embedding_kmeans_ma.csv'
        this_direc = f'data/analysis/{analysis_name}/clusters_single/{clus_type}/ma/'
    if input_data == 'Pa':
        data_path = f'data/analysis/{analysis_name}/clusters_single/{clus_type}/embedding_kmeans_pa.csv'
        this_direc = f'data/analysis/{analysis_name}/clusters_single/{clus_type}/pa/'
    if input_data == 'Dha':
        data_path = f'data/analysis/{analysis_name}/clusters_single/{clus_type}/embedding_kmeans_dha.csv'
        this_direc = f'data/analysis/{analysis_name}/clusters_single/{clus_type}/dha/'
    if input_data == 'Ni':
        data_path = f'data/analysis/{analysis_name}/clusters_single/{clus_type}/embedding_kmeans_ni.csv'
        this_direc = f'data/analysis/{analysis_name}/clusters_single/{clus_type}/ni/'
    if input_data == 'All':
        data_path = f'data/analysis/{analysis_name}/clusters_single/embedding_umap.csv'
        this_direc = ''
        disab = True

    data_paths = []
    svara_level = 'kmeans' in data_path
    title = 'All Svaras' if not svara_level else 'Svara: ' + data_path.split('_')[-1].split('.')[0].capitalize()

    df = pd.read_csv(data_path)
    data_dict['df'] = df

    if svara_level == True:
        label_lookup = {i:f'cluster_{i}' if i != -1 else 'Noise' for i in df['label'].unique()}
        colours = ['#000000','#696969','#8b4513','#006400','#808000','#483d8b','#008b8b','#9acd32','#00008b','#7f007f','#8fbc8f','#b03060','#ff4500','#ffa500','#ffff00','#7cfc00','#deb887','#8a2be2','#00ff7f','#dc143c','#00ffff','#00bfff','#0000ff','#b0c4de','#ff00ff','#1e90ff','#fa8072','#dda0dd','#90ee90','#ff1493']
        random.shuffle(colours)
    else:
        label_lookup = load_pkl(f'data/analysis/{analysis_name}/label_lookup.pkl')
        colours = ['#000000','#66cdaa','#ffa500','#00ff00','#0000ff','#1e90ff','#ff1493']

    x_series = []
    y_series = []
    colours_series = []
    names = []
    pp_series = []
    ap_series= []
    for label in df['label'].unique():
        this_df = df[df['label']==label]
        svara = label_lookup[label]
        x_series.append(this_df['x'].values)
        y_series.append(this_df['y'].values)
        pp_series.append(this_df['plot_path'].values)
        ap_series.append(this_df['audio_path'].values)
        colours_series.append(colours[label])
        names.append(svara)
    
    data_dict['ap_series'] = ap_series
    data_dict['pp_series'] = pp_series

    fig = go.Figure()
    for x, y, c, n in zip(x_series, y_series, colours_series, names):
        fig.add_trace(
                go.Scatter(
                x=x,
                y=y,
                name=n,
                mode='markers',
                marker_color=c,
                marker=dict(
                    line={"color": "#444"},
                    reversescale=True,
                    sizeref=45,
                    sizemode="diameter",
                    opacity=0.8,
                )
            )
        )


    # turn off native plotly.js hover effects - make sure to use
    # hoverinfo="none" rather than "skip" which also halts events.
    fig.update_traces(hoverinfo="none", hovertemplate=None)

    fig.update_layout(
        plot_bgcolor='rgba(255,255,255,0.1)',
        width=700,
        height=700
    )
    
    return fig


if __name__ == "__main__":
    app.run_server(debug=True, port=8010)