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
from sklearn.cluster import AgglomerativeClustering
from sklearn.mixture import GaussianMixture

from sklearn.metrics import silhouette_score
from sklearn.metrics import calinski_harabasz_score
from sklearn.metrics import davies_bouldin_score

from sklearn.preprocessing import MinMaxScaler

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
app = dash.Dash(
__name__, 
)
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
        html.H2("File"),
        dcc.Upload(id='upload_data', children=[html.Button('Upload File')]),
        dcc.Checklist(id='column_names', options=[], labelStyle={'display': 'block'})
        ]),

        #method
        html.Div([
        html.H2("Select clustering method"),
        dcc.Dropdown(id='method', options=[
            {'label': 'Kmeans', 'value': 'Kmeans'},
            {'label': 'Hierarchical', 'value': 'Hierarchical'},
            {'label': 'GMM', 'value': 'GMM'},])
        ]),

        #method params
        html.Div(id='method_params', children=[]),
        html.Br(),

        #run button
        html.Button('Run', id='runbtn', n_clicks=0),

        #scores
        html.Div(id='scores', children=[]),
        html.Br(),

        #export btn
        html.Div([
        html.Button('Export csv', id='export', n_clicks=0),
        dcc.Download(id='download_df')
        ])

    ], style={'width': '20%', 'display': 'inline-block', 'vertical-align': 'top'}),

    #second row
    html.Div([

        #3d viwer
        html.Div([
        dcc.Graph(id='viwer', figure={}),
        
        #proportions plot
        dcc.Graph(id='proportions', figure={})
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
    parameters = [
            html.H3('Parameters'),
        ]
    if mval == "Kmeans":
        parameters.append(dcc.Input(id='nclus', type='number', placeholder='Numb. of clusters', style={'height':'30px'}))
        return parameters
    elif mval == "Hierarchical":
        parameters.append(dcc.Input(id='nclus', type='number', placeholder='Numb. of clusters', style={'height':'30px'}))
        parameters.append(dcc.Input(id='nclus', type='number', placeholder='Dist. treshold', style={'height':'30px'}))
        return parameters
    elif mval == "GMM":
        parameters.append(dcc.Input(id='nclus', type='number', placeholder='Numb. of clusters', style={'height':'30px'}))
        parameters.append(dcc.Dropdown(id='cov_type', options=[
            {'label': 'full', 'value': 'full'},
            {'label': 'tied', 'value': 'tied'},
            {'label': 'diag', 'value': 'diag'},
            {'label': 'spherical', 'value': 'spherical'}],
            placeholder="Covariance type"
        ))
        return parameters
    else:
        return parameters

#run
@app.callback(
[Output('viwer', 'figure'), 
Output('proportions', 'figure'),
Output('scores', 'children')],
[Input('runbtn', 'n_clicks'),
State('column_names', 'value'),
State('method','value'),
State('method_params', 'children')]
)
def run(n_clicks, cols, method, actual_children):
    schildren = [html.H2('Scores')]
    if n_clicks != 0:
        print('Calculating...')
        
        #preparing data
        cols = cols + [x, y, z]
        global dfna
        dfna = df[cols].dropna()
        X = dfna[cols[:-3]].values

        #scaling data
        scaler = MinMaxScaler()
        X = scaler.fit_transform(X)

        #here you will get params and run clustering algorithms
        if method == 'Kmeans':
            nclus = actual_children[1]['props']['value']
            kmeans = KMeans(n_clusters=nclus).fit(X)
            labels = kmeans.labels_
            
            dfna['labels'] = [str(j) for j in labels]
        elif method == 'Hierarchical':
            nclus = actual_children[1]['props']['value']
            dist_tresh = actual_children[2]['props']['value']
            if nclus is None:
               clustering = AgglomerativeClustering(distance_threshold=dist_tresh).fit(X)
            else:
                clustering = AgglomerativeClustering(n_clusters=nclus).fit(X)
            labels = clustering.labels_

            dfna['labels'] = [str(j) for j in labels]
        elif method == 'GMM':
            nclus = actual_children[1]['props']['value']
            cov_type = nclus = actual_children[2]['props']['value']
            gmm = GaussianMixture(n_components=nclus, covariance_type=cov_type)
            labels = gmm.predict(X)

            dfna['labels'] = [str(j) for j in labels]
        else:
            labels = ['0' for j in range(len(dfna))]
            dfna['labels'] = labels
            sc, chs, dbs = 0, 0, 0

        #scores
        sc = silhouette_score(X, labels, metric='euclidean').round(2)
        chs = calinski_harabasz_score(X, labels).round(2)
        dbs = davies_bouldin_score(X, labels).round(2)

        #plotting the 3d scatter
        layout = {
        'height':750,
        'title':'Clusters 3d map',
        }
        scene = dict(
        aspectmode='data',
        aspectratio=dict(x=1, y=1, z=1)
        )

        dfna = dfna.loc[:,~dfna.columns.duplicated()]

        data = []
        for i in dfna['labels'].unique():
            f = dfna['labels'] == i
            
            dfnaf = dfna[f]
            t = go.Scatter3d(x=dfnaf[x], y=dfnaf[y], z=dfnaf[z], mode='markers', marker=dict(size=2), name=i) 
            data.append(t)
      
        fig = go.Figure(data = data, layout = layout)
        fig.layout.scene = scene  

        #ploting poportions
        layout = {
        'height':400,
        'title':'Labels proportions',
        }
        
        l = dfna['labels'].unique()
        h = [len(dfna[dfna['labels'] == i]) for i in l]

        fig1 = go.Figure(data = [go.Bar(x=l, y=h, text=h)], layout = layout)

        #updating scores
        tbl = html.Table(
        #header
        [html.Tr([html.Th('Metric'), html.Th('Score')])] +
        #body
        [html.Tr([html.Td(['Silhouete']), html.Td([str(sc)])])] +
        [html.Tr([html.Td(['Calinski Harabaz']), html.Td([str(chs)])])] +
        [html.Tr([html.Td(['Davies Bouldin']), html.Td([str(dbs)])])]
        )
        
        schildren.append(tbl)
            
        return fig, fig1, schildren
    else:
        return {}, {}, schildren

#exporting
@app.callback(
Output('download_df', 'data'),
Input('export', 'n_clicks')
)
def export(n_clicks):
    if n_clicks != 0:
        return dcc.send_data_frame(dfna.to_csv, 'exported_file.csv')

#----

#----
if __name__ == '__main__':
    app.run_server(debug=True)