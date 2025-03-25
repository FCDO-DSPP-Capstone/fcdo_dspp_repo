import networkx as nx
import pandas as pd
import numpy as np
from dash import Dash, dcc, html, Input, Output
import plotly.graph_objects as go
import dash_table

# Initialize Dash app
app = Dash(__name__)

# File paths for datasets
files = {
    "Semiconductor Silicon Wafer": 'trade_data/TradeData_3_18_2025_16_45_31.csv',
    "Semiconductor Equipment": 'trade_data/TradeData_3_18_2025_16_47_33.csv',
    "Electronic Integrated Circuits": 'trade_data/TradeData_3_18_2025_16_48_22.csv',
    "Electronic Computers and Components": 'trade_data/TradeData_3_18_2025_16_49_41.csv'
}

# Layout
app.layout = html.Div([   
    dcc.Dropdown(
        id="commodity-dropdown",
        style={'font-family': 'Helvetica'},
        options=[{"label": k, "value": v} for k, v in files.items()],
        value=list(files.values())[0],
        clearable=False
    ),
    
    html.Br(),

    dcc.Dropdown(
        id="year-dropdown",
        style={'font-family': 'Helvetica'},
        options=[{"label": str(y), "value": y} for y in [2010, 2020, 2023, 2024]],
        value=2020,
        clearable=False
    ),

    html.Br(),

    dcc.RadioItems(
        id="network-type",
        options=[
            {"label": "Full Network", "value": "full"},
            {"label": "Top 20", "value": "top_20"},
            {"label": "Top 10", "value": "top_10"},
        ],
        value="full",
        labelStyle={'display': 'inline-block', 'margin-right': '10px', 'font-family': 'Helvetica'}
    ),

    dcc.Dropdown(
        id="country-selector",
        options=[],
        placeholder="Select Country to highlight",
        style={'font-family': 'Helvetica'}
    ),

    dcc.Graph(id="network-graph", clear_on_unhover=True, config={'scrollZoom': True}, style={'flex': '1'}),
    
    html.Br(),
    html.Br(),


    html.H3(id='table-title', style={'font-family': 'Helvetica', 'margin-top': '20px'}),
    html.P("This table shows the total exports and total imports expressed in billions of US dollars for the selected year and product", style={'font-family': 'Helvetica', 'font-size': '14px'}),
    
    
    dash_table.DataTable(
        id='trade-table',
        columns=[
            {"name": "Country", "id": "country"},
            {"name": "Total Exports (Billions)", "id": "exports"},
            {"name": "Total Imports (Billions)", "id": "imports"}
        ],
        fixed_rows={'headers': True, 'data': 0},
        style_data={'whiteSpace': 'normal'},
        style_cell={'textAlign': 'left', 'font-family': 'Helvetica', 'overflow': 'hidden', 
        'textOverflow': 'ellipsis', 'maxWidth': 50},
        sort_action="native", 
        style_header={
            'backgroundColor': 'lightgrey',
            'fontWeight': 'bold'
        }
    ),
    
    dcc.Store(id='store-country-selector', data=None)  # Add this line
])

def process_data(file_path, year):
    df = pd.read_csv(file_path, encoding='cp1252')
    df['trade_value'] = df[['cifvalue', 'fobvalue', 'primaryValue']].max(axis=1, skipna=True)
    df = df[df['trade_value'] > 0]
    df = df[df['refPeriodId'] == year]
    excluded_regions = ['World', 'Other Asia, nes', 'Other Europe, nes', 'Other America, nes', 'Special Categories']
    df = df[~df['reporterISO'].isin(excluded_regions) & ~df['partnerISO'].isin(excluded_regions)]
    return df

@app.callback(
    Output("network-graph", "figure"),
    Output("country-selector", "options"),
    Output("store-country-selector", "data"),
    [Input("commodity-dropdown", "value"),
     Input("year-dropdown", "value"),
     Input("network-type", "value"),
     Input("country-selector", "value")]
)
def update_graph(selected_file, selected_year, network_type, selected_country):
    # Load and process the data
    df = process_data(selected_file, selected_year)
    
    # Calculate the top three partners before filtering
    top_partners_dict = {}
    for node in df['reporterISO'].unique():
        trade_partners = df[df['reporterISO'] == node].groupby('partnerISO')['trade_value'].sum().nlargest(3)
        top_partners_dict[node] = trade_partners.index.tolist()

    # Filter network based on user selection
    if network_type in ["top_20", "top_10"]:
        top_n = 20 if network_type == "top_20" else 10
        total_trade = df.groupby('reporterISO')['trade_value'].sum() + df.groupby('partnerISO')['trade_value'].sum()
        top_countries = total_trade.nlargest(top_n).index.tolist()
        df = df[df['reporterISO'].isin(top_countries) & df['partnerISO'].isin(top_countries)]

    # Construct graph
    G = nx.Graph()
    for _, row in df.iterrows():
        G.add_edge(row['reporterISO'], row['partnerISO'], weight=row['trade_value'])

    if len(G.nodes()) == 0:
        return go.Figure(), [], None  # Return updated data

    k_value = 0.3 * (1 / np.sqrt(len(G.nodes()))) if len(G.nodes()) > 0 else 0.1
    pos = nx.spring_layout(G, seed=42, k=k_value)

    hover_texts = []
    node_trade_values = {node: 0 for node in G.nodes()}
    for node in G.nodes():
        top_partners = top_partners_dict.get(node, [])
        total_partners = len(G[node])
        partner_text = ", ".join(top_partners) if top_partners else "None"
        hover_texts.append(f"Country: {node}<br>Total Trade Partners: {total_partners}<br>Top 3 Trade Partners: {partner_text}")
        
        # Calculate trade value for node size
        node_trade_values[node] = df[(df['reporterISO'] == node) | (df['partnerISO'] == node)]['trade_value'].sum()

    edge_x, edge_y = [], []
    node_x, node_y = [], []
    node_colors, node_sizes, hover_infos = [], [], []
    highlighted_edges_x, highlighted_edges_y = [], []

    if selected_country:
        for edge in G.edges(selected_country):
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            highlighted_edges_x.extend([x0, x1, None])
            highlighted_edges_y.extend([y0, y1, None])
    else:
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

    # Determine the node size range
    min_size = 10
    max_size = 30
    max_trade_value = max(node_trade_values.values())
    min_trade_value = min(node_trade_values.values())

    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        if node == selected_country:
            node_colors.append("green")
        elif selected_country and node in G[selected_country]:
            node_colors.append("red")
        else:
            node_colors.append("blue")
        
        # Node size based on trade value, moderated to constrain the size difference
        node_size = min_size + ((node_trade_values[node] - min_trade_value) / (max_trade_value - min_trade_value)) * (max_size - min_size)
        node_sizes.append(node_size)

        hover_infos.append(f"{node}<br>Total Trade Partners: {len(G[node])}<br>Top 3 Trade Partners: {', '.join(top_partners_dict.get(node, []))}")

    fig = go.Figure(data=[
        go.Scatter(x=edge_x, y=edge_y, mode="lines", line=dict(width=0.5, color="gray"), hoverinfo="none", showlegend=False),
        go.Scatter(x=highlighted_edges_x, y=highlighted_edges_y, mode="lines", line=dict(width=0.8, color="red"), hoverinfo="none", showlegend=False),
        go.Scatter(x=node_x, y=node_y, mode="markers", marker=dict(size=node_sizes, color=node_colors), text=hover_infos, hoverinfo="text", showlegend=False)
    ])

    fig.update_layout(template="plotly_white", xaxis=dict(showticklabels=False, ticks="", showgrid=False, zeroline=False),
                      yaxis=dict(showticklabels=False, ticks="", showgrid=False, zeroline=False), plot_bgcolor="rgba(0,0,0,0)")

    country_options = [{"label": country, "value": country} for country in G.nodes()]

    return fig, country_options, selected_country

@app.callback(
    Output('country-selector', 'value'),
    Input('network-type', 'value'),
)
def reset_country_selector(_):
    return None

@app.callback(
    Output('trade-table', 'data'),
    [Input('commodity-dropdown', 'value'), Input('year-dropdown', 'value')]
)
def update_table(selected_file, year):
    df = process_data(selected_file, year)
    
    # Aggregate the data for imports and exports
    imports = (df.groupby('partnerISO')['trade_value'].sum() / 1e9).round(2).reset_index()
    exports = (df.groupby('reporterISO')['trade_value'].sum() / 1e9).round(2).reset_index()
    
    # Merge the dataframes on country names
    merged_df = pd.merge(imports, exports, left_on='partnerISO', right_on='reporterISO', how='outer', suffixes=('_imports', '_exports'))
    merged_df = merged_df.fillna(0)  # Fill NaN values with 0
    
    # Rename columns for clarity
    merged_df = merged_df.rename(columns={
        'partnerISO': 'country',
        'trade_value_imports': 'imports',
        'trade_value_exports': 'exports'
    })
    
    merged_df = merged_df.sort_values(by=['exports'], ascending=False)
    return merged_df[['country', 'imports', 'exports']].to_dict('records')


@app.callback(
    Output('table-title', 'children'),
    [Input('commodity-dropdown', 'value'), Input('year-dropdown', 'value')]
)
def update_table_title(selected_file, selected_year):
    product_name = [key for key, value in files.items() if value == selected_file][0]
    return f"Table for {product_name} in {selected_year}"


if __name__ == "__main__":
    app.run_server(debug=True)