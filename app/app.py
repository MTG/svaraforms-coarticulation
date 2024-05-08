import os
import pickle
import sys

from dash import Dash, dcc, html, Input, Output, callback
import plotly.graph_objects as go
import plotly.express as px

from visualisation import get_plot_kwargs
from utils import load_pkl

# cpath('app','audio',f'{raga}',f'{svara}',f'{occurrence}.wav')

raga = 'bhairavi'

svara_dict = load_pkl(os.path.join('data', 'svara_dict', f'svara_dict_{raga}.pkl'))
unique_svaras = list(svara_dict.keys())

app = Dash(__name__)
app.title = 'Svara Explorer'
app._favicon = ("/assets/favicon.ico")

dd_style = {'width': '50%'}

app.layout = html.Div([

    html.H1(f'Svara explorer for raga, {raga}'),
    html.Div([
        html.Div([
            html.Div([
                "Svara",
                dcc.Dropdown(
                    unique_svaras,
                    placeholder="Select a svara",
                    id='svara-dropdown',
                    style=dd_style
                    )
                ]),

            html.Div([
                "Occurrence",
                dcc.Dropdown(
                    placeholder="Select an occurrence",
                    id='occurrence-dropdown',
                    style=dd_style
                    )
                ])
            ]),
        html.Div([html.Audio(id='audio-path', controls=True)],style={"margin-top": "15px"}),
    ]),

    dcc.Graph(id='graph')]
)

@app.callback(
    Output('occurrence-dropdown', 'options'),
    [Input('svara-dropdown', 'value')]
)
def update_occ_dropdown(svara):
    if svara != None:
        return [{'label': i, 'value': i} for i in range(len(svara_dict[svara]))]
    else:
        return []

@app.callback(Output('audio-path', 'src'),
              [Input('svara-dropdown', 'value'), Input('occurrence-dropdown', 'value')])
def get_audio_path(svara, occurrence):
    if svara != None and occurrence != None:
        return os.path.join('/assets','audio', f'{raga}',f'{svara}',f'{occurrence}.wav')
    else:
        return ''


@app.callback(Output('graph', 'figure'),
              [Input('svara-dropdown', 'value'), Input('occurrence-dropdown', 'value')])
def update_figure(svara, occurrence):
    ready = svara != None and occurrence != None
    if ready:
        i=occurrence

        svaras = svara_dict[svara]

        d = svaras[i]

        track = d['track']
        timestep = d['timestep']
        y = d['pitch']
        tonic = d['tonic']
        x = [d['start'] + i*timestep for i in range(len(y))]
        
        title = f"Occurrence {i} of {svara} in recording, {track}"
        ytitle = f'Pitch (cents above tonic of {tonic}Hz)'

        plot_kwargs = get_plot_kwargs(raga, tonic)
        yticks_dict = plot_kwargs['yticks_dict']

    else:
        i=10
        title = ''
        ytitle = f'Pitch (cents)'
        y = []
        x = []

    layout = go.Layout(
        autosize=False,
        width=1200,
        height=600,
        xaxis=go.layout.XAxis(linecolor="black", linewidth=1, mirror=True),
        yaxis=go.layout.YAxis(linecolor="black", linewidth=1, mirror=True),
        margin=go.layout.Margin(l=25, r=25, b=100, t=100, pad=4),
    )

    fig = go.Figure([go.Scatter(x=x, y=y)], layout=layout)
    
    if ready:
        fig.update_yaxes(tickvals=list(yticks_dict.values()), ticktext=list(yticks_dict.keys()))

    fig.update_layout(
        title=dict(text=title, font=dict(size=20), yref='paper'),
        xaxis_title='Time (s)',
        yaxis_title=ytitle
    )

    return fig

if __name__ == '__main__':
    app.run(debug=True)
