import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table
from dash.dependencies import Input, Output
import pandas as pd

# File paths for datasets
files = {
    "Semiconductor Silicon Wafer": 'trade_data/TradeData_3_18_2025_16_45_31.csv',
    "Semiconductor Equipment": 'trade_data/TradeData_3_18_2025_16_47_33.csv',
    "Electronic Integrated Circuits": 'trade_data/TradeData_3_18_2025_16_48_22.csv',
    "Electronic Computers and Components": 'trade_data/TradeData_3_18_2025_16_49_41.csv'
}

def process_data(file_path):
    df = pd.read_csv(file_path, encoding='cp1252')
    df['trade_value'] = df[['cifvalue', 'fobvalue', 'primaryValue']].max(axis=1, skipna=True)
    df = df[df['trade_value'] > 0]
    
    # Rename "China, Hong Kong SAR" to "Hong Kong SAR"
    df.replace({'reporterISO': {'China, Hong Kong SAR': 'Hong Kong SAR'},
                'partnerISO': {'China, Hong Kong SAR': 'Hong Kong SAR'}}, inplace=True)
    
    # Expanded list of excluded non-country regions
    excluded_regions = [
        'World', 'Other Asia, nes', 'Other Europe, nes', 
        'Other America, nes', 'Special Categories', 
        'Unspecified', 'Areas, nes'
    ]
    
    df = df[~df['reporterISO'].isin(excluded_regions) & ~df['partnerISO'].isin(excluded_regions)]
    
    return df

def get_top_trade(df, year):
    df = df[df['refPeriodId'] == year]
    top_exports = df.groupby('reporterISO')['trade_value'].sum().reset_index()
    top_imports = df.groupby('partnerISO')['trade_value'].sum().reset_index()
    
    # Convert to **billions** instead of millions
    top_exports['Total Exports'] = (top_exports['trade_value'] / 1e9).round(2)
    top_imports['Total Imports'] = (top_imports['trade_value'] / 1e9).round(2)
    
    top_exports = top_exports[['reporterISO', 'Total Exports']]
    top_imports = top_imports[['partnerISO', 'Total Imports']]
    
    # Update column names to reflect billions
    top_exports.columns = ['Country', 'Total Exports (Billions)']
    top_imports.columns = ['Country', 'Total Imports (Billions)']
    
    # Merge the two tables
    merged_table = pd.merge(top_exports, top_imports, on='Country', how='outer').fillna(0).sort_values(by=['Total Imports (Billions)'], ascending=False)
    
    return merged_table

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H3('Trade Data Tables', style={'font-family': 'Helvetica'}),
    dcc.Dropdown(
        id='product-dropdown',
        options=[{'label': product, 'value': product} for product in files.keys()],
        value='Semiconductor Silicon Wafer',
        style={'font-family': 'Helvetica'}
    ),
    dcc.Dropdown(
        id='year-dropdown',
        options=[2010],
        value=2010, 
        style={'font-family': 'Helvetica'}
    ),
    dash_table.DataTable(
        id='trade-table',
        columns=[{"name": i, "id": i} for i in ['Country', 'Total Imports (Billions)', 'Total Exports (Billions)']],
        fixed_rows={'headers': True, 'data': 0},
        style_data={'whiteSpace': 'normal'},
        style_cell={'textAlign': 'left', 'font-family': 'Helvetica', 'overflow': 'hidden', 
        'textOverflow': 'ellipsis', 'maxWidth': 50},
        sort_action="native", 
        style_header={
            'backgroundColor': 'lightgrey',
            'fontWeight': 'bold'

        }
    )
])

@app.callback(
    Output('year-dropdown', 'options'),
    [Input('product-dropdown', 'value')]
)
def update_year_options(product):
    df = process_data(files[product])
    years = sorted(df['refPeriodId'].unique())
    return [{'label': year, 'value': year} for year in years]

@app.callback(
    Output('trade-table', 'data'),
    [Input('product-dropdown', 'value'), Input('year-dropdown', 'value')]
)
def update_table(product, year):
    df = process_data(files[product])
    merged_table = get_top_trade(df, year)
    return merged_table.to_dict('records')

if __name__ == "__main__":
    app.run_server(debug=True)

