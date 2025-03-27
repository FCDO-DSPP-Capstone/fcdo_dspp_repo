### UN voting process source code ### 
""" This scripts imports the results from the PCA and clustering of UN votes to 
make an app with plots, table and more information
It also enrich the data set with world bank data for population and GDP

"""
####################################
## 0. PACKAGES ##################
####################################

import pandas as pd
import numpy as np
import plotly.express as px

import dash
from dash import dcc, html, Input, Output, dash_table
import pandas as pd
from dash.dash_table import DataTable, FormatTemplate

from sklearn.metrics import pairwise_distances

####################################
## 1. IMPORT DATA ##################
####################################
pca_results = pd.read_csv('Juan - UN votes/data_output/pca_results.csv')
pca_results = pca_results[pca_results['year'] >= 1990]

pca_results['cluster'] = pca_results['cluster'].astype('category') 

association_scores = pd.read_csv('Juan - UN votes/data_output/association_scores.csv') 

mean_distance_table = association_scores.groupby('Country 1')['Mean distance'].mean().reset_index()
mean_distance_table = mean_distance_table.rename(columns={'Mean distance': 'Mean Mean Distance'})



#######################################################
## 2. UN PCA votes plot with year slider ##
#######################################################

#### Combined app testing 
app = dash.Dash(__name__)

percentage = FormatTemplate.percentage(2)

app.layout = html.Div([
    html.H1('UN votes in the General Assembly', style={'font-family': 'Helvetica', 'text-align': 'center'}),
    html.H4('Nominal GDP per capita percentile', style={'font-family': 'Helvetica'}),
    dcc.RangeSlider(
        id='gdp-slider',
        min=0,
        max=1,
        value=[0, 1],
        marks={str(i): {'label': str(int(i * 100)) + '%', 'style': {'font-family': 'Helvetica'}} for i in np.arange(0, 1.01, 0.05)},
    ),
    html.H4('Population percentile', style={'font-family': 'Helvetica'}),
    dcc.RangeSlider(
        id='pop-slider',
        min=0,
        max=1,
        value=[0, 1],
        marks={str(i): {'label': str(int(i * 100)) + '%', 'style': {'font-family': 'Helvetica'}} for i in np.arange(0, 1.01, 0.05)},
    ),

    html.Br(),
    dcc.Graph(id='pca-graph', style={'font-family': 'Helvetica', 'height': '600px'}),

    html.H4('Select year', style={'font-family': 'Helvetica'}),
    dcc.Slider(
        id='year-slider',
        min=pca_results['year'].min(),
        max=pca_results['year'].max(),
        value=pca_results['year'].max(),
        marks={str(i): {'label': str(i), 'style': {'font-family': 'Helvetica', 'writing-mode': 'vertical-lr'}} for i in range(pca_results['year'].min(), pca_results['year'].max(), 1)},
        step=1
    ),

    html.Div(style={'height': '100px'}),

    html.H3("Country Pairwise Co-Cluster Table since 1990", style={'font-family': 'Helvetica'}),
    dcc.Dropdown(
        id='country-1-dropdown',
        value='United Kingdom',
        options=[{'label': c, 'value': c} for c in sorted(association_scores['Country 1'].unique())],
        clearable=False,
        style={'font-family': 'Helvetica'}
    ),
    dash_table.DataTable(
        id='filtered-table',
        columns=[
            {'name': 'Country 2', 'id': 'Country 2'},
            dict(name='Co-Cluster Score', id='Co-Cluster Score', type='numeric', format=percentage),
            dict(name='Mean distance', id='Mean distance', type='numeric', format=percentage)
        ],
        style_table={'overflowX': 'auto'},
        style_cell={'textAlign': 'left', 'font-family': 'Helvetica'},
        sort_action="native"
    ), 
    
    html.Br(), 

    html.H3("Mean Mean Distance Table", style={'font-family': 'Helvetica'}),
    dash_table.DataTable(
        id='mean-distance-table',
        columns=[
            {'name': 'Country 1', 'id': 'Country 1'},
            dict(name='Mean Mean Distance', id='Mean Mean Distance', type='numeric', format=percentage)
        ],
        data=mean_distance_table.to_dict('records'),
        style_table={'overflowX': 'auto'},
        style_cell={'textAlign': 'left', 'font-family': 'Helvetica'},
        sort_action="native"
    )
])

@app.callback(
    Output('pca-graph', 'figure'),
    Input('gdp-slider', 'value'),
    Input('pop-slider', 'value'),
    Input('year-slider', 'value')
)
def update_figure(gdp_range, pop_range, year):
    pca_results_filtered = pca_results[(pca_results['gdp_pp'] >= gdp_range[0]) & (pca_results['gdp_pp'] <= gdp_range[1]) & (pca_results['pop'] >= pop_range[0]) & (pca_results['pop'] <= pop_range[1]) & (pca_results['year'] == year)]
    fig = px.scatter(
        pca_results_filtered,
        x='PCA1',
        y='PCA2',
        color='region_name',
        color_discrete_sequence=['rgb(17, 112, 170)', 'rgb(252, 125, 11)', 'rgb(163, 172, 185)', 'rgb(95, 162, 206)', 'rgb(200, 82, 0)', 'rgb(123, 132, 143)', 'rgb(163, 204, 233)', 'rgb(255, 188, 121)', 'rgb(200, 208, 217)'],
        hover_data={'ms_name': True, 
                    'region_name': True,
                     'PCA1': False, 'PCA2': False, 'pop': ':.2%', 'gdp_pp': ':.2%', 'cluster': False},
        labels={'PCA1': '', 'PCA2': ''},
        template='plotly_white'
    )
    fig.update_layout(
        font=dict(
            family='Helvetica',
            size=14,
            color='black'
        ),
        title_x=0.5,
        title={
            'text': f'UN countries relative position given their votes',
            'subtitle': {
                'text': f'Year: {year} <br> GDP PP range: {gdp_range[0]:.2%} to {gdp_range[1]:.2%} <br> Population range: {pop_range[0]:.2%} to {pop_range[1]:.2%} <br> Total countries: {len(pca_results_filtered)} <br> ',
            }
        }, 
        margin = dict(t = 200),
        xaxis=dict(
            showticklabels=False,
        ),
        yaxis=dict(
            showticklabels=False,
        ),
        legend_title_text = 'Continent',
    )
    return fig

@app.callback(
    Output('filtered-table', 'data'),
    Input('country-1-dropdown', 'value'),
)
def update_table(selected_country):
    filtered_df = association_scores[association_scores['Country 1'] == selected_country].copy()
    return filtered_df.to_dict('records')

if __name__ == '__main__':
    app.run(debug=True)

#######




# Calculate the correlation between Co-Cluster Score and Mean Distance
#correlation = association_scores[['Co-Cluster Score', 'Mean distance']].corr().iloc[0, 1]

# Print the total correlation
#print(f"Total Correlation between Co-Cluster Score and Mean Distance: {correlation:.2f}")


### Separate votes by country # most impurtant votes?
## Agregarle botón para seleccionar que países resaltar desde la lista completa para any givren year
# eso hay q hacerlo en das

# Mezclar todas las aplicaciones 

