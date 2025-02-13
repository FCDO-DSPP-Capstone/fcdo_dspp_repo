import plotly.graph_objects as go

# Example node data with categories
nodes = [
    {"id": "Node A", "x": 1, "y": 2, "category": "Group 1"},
    {"id": "Node B", "x": 2, "y": 3, "category": "Group 1"},
    {"id": "Node C", "x": 3, "y": 1, "category": "Group 2"},
    {"id": "Node D", "x": 4, "y": 2, "category": "Group 2"},
    {"id": "Node E", "x": 5, "y": 3, "category": "Group 3"}
]

edges = [("Node A", "Node B"), ("Node B", "Node C"), ("Node C", "Node D"), ("Node D", "Node E")]

# Create node positions
node_x, node_y, node_labels, node_categories = [], [], [], []
for node in nodes:
    node_x.append(node["x"])
    node_y.append(node["y"])
    node_labels.append(node["id"])
    node_categories.append(node["category"])

# Create edge traces
edge_x, edge_y = [], []
for edge in edges:
    start = next(node for node in nodes if node["id"] == edge[0])
    end = next(node for node in nodes if node["id"] == edge[1])
    edge_x.extend([start["x"], end["x"], None])
    edge_y.extend([start["y"], end["y"], None])

edge_trace = go.Scatter(
    x=edge_x, y=edge_y,
    line=dict(width=0.5, color="gray"),
    hoverinfo="none",
    mode="lines"
)

# Create node trace
node_trace = go.Scatter(
    x=node_x, y=node_y,
    mode="markers+text",
    marker=dict(size=10, color="blue", line_width=2),
    text=node_labels,
    textposition="top center",
    hoverinfo="text"
)

# Dropdown menu for category selection
categories = list(set(node_categories))
buttons = [
    dict(
        label=cat,
        method="update",
        args=[
            {"marker.color": ["blue" if c == cat else "lightgray" for c in node_categories]},
            {"title": f"Network Graph - {cat}"}
        ]
    ) for cat in categories
]

fig = go.Figure(
    data=[edge_trace, node_trace],
    layout=go.Layout(
        title="Network Graph with Node Filtering",
        showlegend=False,
        hovermode="closest",
        margin=dict(b=20, l=5, r=5, t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        updatemenus=[dict(buttons=buttons, direction="down", showactive=True)]
    )
)

fig.show()