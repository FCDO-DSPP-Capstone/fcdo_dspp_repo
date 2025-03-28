### UN voting process source code ### 
""" This scripts imports the results from the PCA  to make an app 
with plots, tables and more information.
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

####################################
## 1. IMPORT DATA ##################
####################################
pca_results = pd.read_csv('data_in/pca_results.csv')
pca_results = pca_results[pca_results['year'] >= 1990]

pca_results['cluster'] = pca_results['cluster'].astype('category') 

association_scores = pd.read_csv('data_in/association_scores.csv') 

mean_distance_table = association_scores.groupby('Country 1')['Mean distance'].mean().reset_index()
mean_distance_table = mean_distance_table.rename(columns={'Mean distance': 'Mean Mean Distance'})

unique_regions = pca_results['region_name'].unique()


colors = ['rgb(17, 112, 170)', 'rgb(252, 125, 11)', 'rgb(163, 172, 185)', 'rgb(95, 162, 206)', 'rgb(200, 82, 0)', 'rgb(123, 132, 143)', 'rgb(163, 204, 233)', 'rgb(255, 188, 121)', 'rgb(200, 208, 217)']
color_mapping = {region: color for region, color in zip(unique_regions, colors[:len(unique_regions)])}

#######################################################
## 2. UN PCA votes plot ##
#######################################################

#### Combined app testing 
app = dash.Dash(__name__)

percentage = FormatTemplate.percentage(2)

app.layout = html.Div([

        ## Year selector slider
    html.H5('Select year', style={'font-family': 'Helvetica'}),
    dcc.Slider(
        id='year-slider',
        min=pca_results['year'].min(),
        max=pca_results['year'].max(),
        value=pca_results['year'].max(),
        marks={str(i): {'label': str(i), 'style': {'font-family': 'Helvetica', 'writing-mode': 'vertical-lr'}} for i in range(pca_results['year'].min(), pca_results['year'].max()+1, 1)},
        step=1,
        included=False
    ),
    html.Div(style={'height': '5px'}),

    ## PCA GRAPH
    dcc.Graph(
        id='pca-graph',
        style={
            'font-family': 'Helvetica',
            'height': '600px',
            'flex': '1',
            'margin-bottom': '-20px'  # Added lower margin
        },
        config={'scrollZoom': True}
    ),


        ## SLIDER TITLE (gdp)
    html.H5('Nominal GDP per capita percentile', style={'font-family': 'Helvetica'}),
    dcc.RangeSlider(
        id='gdp-slider',
        min=0,
        max=1,
        value=[0, 1],
        marks={str(i): {'label': str(int(i * 100)) + '%', 'style': {'font-family': 'Helvetica'}} for i in np.arange(0, 1.01, 0.05)},
    ),
    ## SLIDER TITLE (population)
    html.H5('Population percentile', style={'font-family': 'Helvetica'}),
    dcc.RangeSlider(
        id='pop-slider',
        min=0,
        max=1,
        value=[0, 1],
        marks={str(i): {'label': str(int(i * 100)) + '%', 'style': {'font-family': 'Helvetica'}} for i in np.arange(0, 1.01, 0.05)},
    ),

    html.Div(style={'height': '100px'}),

    ## Table of relations beteween countries based on the PCA
        # first, the selector of country button 
    html.H3("Country Pairwise mean distance 1990 - 2024", style={'font-family': 'Helvetica'}),
    dcc.Dropdown(
        id='country-1-dropdown',
        value='United Kingdom',
        options=[{'label': c, 'value': c} for c in sorted(association_scores['Country 1'].unique())],
        clearable=False,
        style={'font-family': 'Helvetica'}
    ),
      html.Br(),
    
    ## then the table it self
    dash_table.DataTable(
        id='filtered-table',
        columns=[
            {'name': 'Country', 'id': 'Country 2'},
            dict(name='Mean distance', id='Mean distance', type='numeric', format=percentage)
        ],
        fixed_rows={'headers': True, 'data': 0},
        style_table={'overflowX': 'auto',
                'maxHeight': '300px'},
        style_cell={'textAlign': 'left', 'font-family': 'Helvetica', 'overflow': 'hidden', 
        'textOverflow': 'ellipsis', 'maxWidth': 50},
        sort_action="native", 
        style_header={
            'backgroundColor': 'lightgrey',
            'fontWeight': 'bold'

        }
    ), 
    
    html.Br(), 

    ## Table with the mean of distance of a country with all the rest. 
    html.H3("Country mean distance from all the rest 1990 - 2024", style={'font-family': 'Helvetica'}),
    dash_table.DataTable(
        id='mean-distance-table',
        
        columns=[
            {'name': 'Country', 'id': 'Country 1'},
            dict(name='Average mean distance', id='Mean Mean Distance', type='numeric', format=percentage)
        ],
        fixed_rows={'headers': True, 'data': 0},
        data=mean_distance_table.to_dict('records'),
        style_table={'overflowX': 'auto',
                'maxHeight': '300px'},
        style_cell={'textAlign': 'left', 'font-family': 'Helvetica', 'overflow': 'hidden', 'textOverflow': 'ellipsis', 'maxWidth': 50, 'minWidth': 50},
        sort_action="native",
        style_header={
            'backgroundColor': 'lightgrey',
            'fontWeight': 'bold'
        }
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
        color_discrete_map=color_mapping,
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
            'text': f'UN Countries: Relative Positions by Voting Patterns',
            'subtitle': {
                'text': f'Year: {year} <br> GDP PP range: {gdp_range[0]:.2%} to {gdp_range[1]:.2%} <br> Population range: {pop_range[0]:.2%} to {pop_range[1]:.2%} <br> Total countries: {len(pca_results_filtered)} <br> ',
            }
        }, 
        margin = dict(t = 200, l = 0),
        xaxis=dict(
            showticklabels=False,
        ),
        yaxis=dict(
            showticklabels=False,
        ),
        legend_title_text = 'Continent',
        uirevision='constant'
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
