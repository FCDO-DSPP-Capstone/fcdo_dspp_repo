import networkx as nx
import pandas as pd
import numpy as np
from dash import Dash, dcc, html, Input, Output
import plotly.graph_objects as go

# Initialize Dash app
app = Dash(__name__)

# File paths for datasets
files = {
    "Semiconductor Silicon Wafer": 'web_page/apps/comtrade/trade_data/TradeData_3_18_2025_16_45_31.csv',
    "Semiconductor Equipment": 'web_page/apps/comtrade/trade_data/TradeData_3_18_2025_16_47_33.csv',
    "Electronic Integrated Circuits": 'web_page/apps/comtrade/trade_data/TradeData_3_18_2025_16_48_22.csv',
    "Electronic Computers and Components": 'web_page/apps/comtrade/trade_data/TradeData_3_18_2025_16_49_41.csv'
}

# Define node colors for each trade category
category_colors = {
    "Semiconductor Silicon Wafer": "blue",
    "Semiconductor Equipment": "green",
    "Electronic Integrated Circuits": "red",
    "Electronic Computers and Components": "purple"
}

# Layout
app.layout = html.Div([
    html.H1("Global Trade Network"),

    dcc.Dropdown(
        id="commodity-dropdown",
        options=[{"label": k, "value": v} for k, v in files.items()],
        value=list(files.values())[0],
        clearable=False
    ),

    dcc.RadioItems(
        id="network-type",
        options=[
            {"label": "Full Network", "value": "full"},
            {"label": "Top 20", "value": "top_20"},
            {"label": "Top 10", "value": "top_10"},
            
        ],
        value="full",
        labelStyle={'display': 'inline-block', 'margin-right': '10px'}
    ),

    dcc.Dropdown(
        id="year-dropdown",
        options=[{"label": str(y), "value": y} for y in [2010, 2020, 2023, 2024]],
        value=2024,
        clearable=False
    ),

    dcc.Dropdown(
        id="label-selection",
        options=[],
        multi=True,
        placeholder="Select countries to display labels"
    ),

    html.Div([
        dcc.Graph(id="network-graph", clear_on_unhover=True, config={'scrollZoom': True}, style={'flex': '1'}),
        html.Div(id="hovered-country-pairings-container", children=[
            html.H3(id="hovered-country-name", children="Top 3 Trade Partners"),
            html.Ul(id="hovered-country-pairings")
        ], style={'position': 'absolute', 'top': '10px', 'right': '10px', 'background-color': 'white', 'padding': '10px', 'border': '1px solid black', 'z-index': '10'})
    ], style={'display': 'flex', 'position': 'relative'})
])

@app.callback(
    [Output("network-graph", "figure"),
     Output("hovered-country-name", "children"),
     Output("hovered-country-pairings", "children"),
     Output("label-selection", "options")],
    [Input("commodity-dropdown", "value"),
     Input("year-dropdown", "value"),
     Input("network-type", "value"),
     Input("network-graph", "hoverData"),
     Input("label-selection", "value")]
)
def update_graph(selected_file, selected_year, network_type, hoverData, selected_labels):
    if selected_labels is None:
        selected_labels = []

    # Load the data
    df = pd.read_csv(selected_file, encoding='cp1252')

    # Extract the selected category and assign colors
    category_name = [key for key, value in files.items() if value == selected_file][0]
    node_color = category_colors.get(category_name, "gray")  # Default to gray if not found

    # Ensure 'refPeriodId' exists and extract the year
    if 'refPeriodId' not in df.columns:
        raise KeyError("The dataset does not contain a 'refPeriodId' column.")

    df['Year'] = df['refPeriodId']  # Use this for filtering

    # Filter data by the selected year
    df = df[df['Year'] == selected_year]

    # Create 'trade_value' column
    df['trade_value'] = df[['cifvalue', 'fobvalue', 'primaryValue']].max(axis=1, skipna=True)

    # Remove rows with missing or zero trade values
    df = df[df['trade_value'] > 0]

    # Apply network filtering
    if network_type == "imports":
        df = df[df['flowDesc'] == "Import"]
    elif network_type == "exports":
        df = df[df['flowDesc'] == "Export"]
    elif network_type in ["top_20", "top_10"]:
        top_n = 20 if network_type == "top_20" else 10
        total_trade = df.groupby('reporterISO')['trade_value'].sum() + df.groupby('partnerISO')['trade_value'].sum()
        top_countries = total_trade.nlargest(top_n).index.tolist()
        df = df[df['reporterISO'].isin(top_countries) & df['partnerISO'].isin(top_countries)]

    # Exclude aggregate regions
    excluded_regions = [
        'World', 'Other Asia, nes', 'Other Europe, nes', 
        'Other America, nes', 'Special Categories', 
        'Unspecified', 'Areas, nes'
    ]
    df = df[~df['reporterISO'].isin(excluded_regions) & ~df['partnerISO'].isin(excluded_regions)]

    # Construct trade network
    G = nx.Graph()
    for _, row in df.iterrows():
        G.add_edge(row['reporterISO'], row['partnerISO'], weight=row['trade_value'])

    # Handle empty graph case
    if len(G.nodes()) == 0:
        return go.Figure(), "Top 3 Trade Partners", [html.Li("None")], []

    # Position nodes using the spring layout
    k_value = 0.3 * (1 / np.sqrt(len(G.nodes()))) if len(G.nodes()) > 0 else 0.1
    pos = nx.spring_layout(G, seed=42, k=k_value)

    # Create dropdown options for country selection
    country_options = [{"label": country, "value": country} for country in G.nodes()]

    # Get hovered country
    hovered_node = None
    if hoverData and "points" in hoverData and hoverData["points"]:
        if "text" in hoverData["points"][0]:
            hovered_node = hoverData["points"][0]["text"].split(": ")[-1]

    # Find top 3 trade partners
    hovered_pairings = []
    hovered_country_label = "Top 3 Trade Partners"
    hover_texts = []
    for node in G.nodes():
        trade_partners = sorted(G[node].items(), key=lambda x: x[1]['weight'], reverse=True)
        top_partners = [partner[0] for partner in trade_partners[:3]]
        total_partners = len(G[node])  # Count total trade partners
        partner_text = ", ".join(top_partners) if top_partners else "None"

        hover_texts.append(f"Country: {node}<br>Total Trade Partners: {total_partners}<br>Top 3 Trade Partners: {partner_text}")

        if node == hovered_node:
            hovered_country_label = f"Top 3 Trade Partners for {hovered_node}"
            hovered_pairings = [html.Li(f"{p}") for p in top_partners]

    # Prepare graph elements
    edge_x, edge_y = [], []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode="lines", line=dict(width=0.5, color="gray"), hoverinfo="none"))
    fig.add_trace(go.Scatter(x=[pos[node][0] for node in G.nodes()], y=[pos[node][1] for node in G.nodes()],
                             mode="markers", marker=dict(size=10, color=node_color),
                             text=hover_texts, hoverinfo="text"))

    return fig, hovered_country_label, hovered_pairings if hovered_pairings else [html.Li("None")], country_options

if __name__ == "__main__":
    app.run_server(debug=True)





