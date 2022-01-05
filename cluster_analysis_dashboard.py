#importing packages
import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go

import pandas as pd

import base64
import io

from sklearn.cluster import KMeans

#----
#defining helper functions
def parse_contents(contents, filename, date):
    #read upload content to a dataframe    
    
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    
    df = pd.read_csv(io.StringIO(decoded.decode('utf-8')), na_values=nanvalue, sep=',', low_memory=False)

    return df
#----

#----
#defining column names and parameters
x = 'LOCATIONX'
y = 'LOCATIONY'
z = 'LOCATIONZ'
nanvalue = -99
#----

#starting the app
app = dash.Dash(__name__)
app.config.suppress_callback_exceptions=True

#defining the app layout
app.layout = html.Div([

    #title
    html.Div([
    html.Img(src='assets/alcoa_logo.png', style={'height':'50px', 'display':'inline-block', 'vertical-align': 'middle'}),
    html.H1("Cluster analysis dashboard", style={'display':'inline-block', 'vertical-align': 'middle', 'padding':'0px 50px'}),
    ], style={}),

    #first row
    html.Div([

        #file
        html.Div([
        dcc.Upload(id='upload_data', children=[html.Button('Upload File')]),
        dcc.Checklist(id='column_names', options=[], labelStyle={'display': 'block'})
        ]),

        #method
        html.Div([
        dcc.Dropdown(id='method', options=[
            {'label': 'Kmeans', 'value': 'Kmeans'},
            {'label': 'GMM', 'value': 'GMM'},])
        ]),

        #method params
        html.Div(id='method_params', children=[]),

        #run button
        html.Button('Run', id='runbtn', n_clicks=0)

    ], style={'width': '20%', 'display': 'inline-block', 'vertical-align': 'top'}),

    #second row
    html.Div([

        #3d viwer
        html.Div([
        dcc.Graph(id='viwer', figure={})
        ]),

    ], style={'width': '80%', 'display': 'inline-block', 'vertical-align': 'top'})
])

#----
#connecting stuff
#upload file 
@app.callback(
Output('column_names', 'options'),
Input('upload_data', 'contents'),
State('upload_data', 'filename'),
State('upload_data', 'last_modified')
)
def read_file(contents, filename, date):

    if contents is not None:
        print("File uploaded: {}".format(filename))

        #reading in a DataFrame
        global df
        df = parse_contents(contents, filename, date)
        
        #updating checkboxes with variables
        cols = list(df.columns)
        val_options = [{'label': x, 'value' : x } for x in cols]
        
        return val_options
    
    else:
        return []

#add method params
@app.callback(
Output('method_params', 'children'),
Input('method', 'value'),
State('method_params', 'children'),
)
def method_params(mval, actual_children):
    if mval == "Kmeans":
        parameters = [
            dcc.Input(id='nclus', type='number', placeholder='Numb. of clusters')
        ]
        return parameters
    elif mval == "GMM":
        return []
    else:
        return []

#run
@app.callback(
Output('viwer', 'figure'),
Input('runbtn', 'n_clicks'),
State('column_names', 'value'),
State('method_params', 'children')
)
def run(n_clicks, cols, actual_children):
    if n_clicks != 0:
        print('Calculating...')
        nclus = actual_children[0]['props']['value']
        
        #runnign kmeans
        cols = cols + [x, y, z]
        dfna = df[cols].dropna()
        X = dfna[cols[:-3]].values
        kmeans = KMeans(n_clusters=nclus).fit(X)
        labels = kmeans.labels_
        dfna['labels'] = [str(j) for j in labels]

        #plotting the 3d scatter
        layout = {
        'height':900,
        'title':'Clusters 3d map',
        }
        scene = dict(
        aspectmode='data',
        aspectratio=dict(x=1, y=1, z=1)
        )

        data = []
        for i in dfna['labels'].unique():
            f = dfna['labels'] == i
            dfnaf = dfna[f]
            dfnaf = dfnaf.loc[:,~dfnaf.columns.duplicated()]
            t = go.Scatter3d(x=dfnaf[x], y=dfnaf[y], z=dfnaf[z], mode='markers', marker=dict(size=5), name=i) 
            data.append(t)
      
        fig = go.Figure(data = data, layout = layout)
        fig.layout.scene = scene  

        return fig
    else:
        return {}
#----

#----
if __name__ == '__main__':
    app.run_server(debug=True)