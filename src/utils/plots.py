import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import pandas as pd
import plotly.graph_objects as go

import numpy as np

class FraudMap:
    def __init__(self, fraud_transactions):
        self.fraud_transactions = fraud_transactions
        # Clean up the credit card numbers
        self.fraud_transactions['cc_num'] = self.fraud_transactions['cc_num'].astype(str).str.strip()

        # Prepare adjusted coordinates for points
        self.fraud_transactions = self.add_flower_pattern(self.fraud_transactions)

        self.app = dash.Dash(__name__)
        self.fig = self.create_map_figure(show_cardholders=True, show_merchants=False)
        self.setup_layout()
        self.setup_callbacks()

    def add_flower_pattern(self, transactions):
        # Group by lat and long
        grouped = transactions.groupby(['lat', 'long'])

        # Add offsets for points at the same location
        new_coords = []
        for (lat, long), group in grouped:
            n = len(group)
            if n > 1:
                # Generate "flower" pattern offsets
                angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
                radius = 0.001  # Adjust radius as needed
                lat_offsets = radius * np.sin(angles)
                long_offsets = radius * np.cos(angles)

                # Apply offsets
                for i, row in enumerate(group.itertuples()):
                    new_coords.append((row.Index, lat + lat_offsets[i], long + long_offsets[i]))
            else:
                # No adjustment needed
                row = group.iloc[0]
                new_coords.append((row.name, lat, long))

        # Create new dataframe with adjusted coordinates
        coords_df = pd.DataFrame(new_coords, columns=['index', 'lat_adjusted', 'long_adjusted']).set_index('index')
        return transactions.join(coords_df)

    def create_map_figure(self, show_cardholders, show_merchants):
        fig = go.Figure()

        if show_cardholders:
            # Cardholders layer with adjusted coordinates
            fig.add_trace(go.Scattermapbox(
                lat=self.fraud_transactions["lat_adjusted"],
                lon=self.fraud_transactions["long_adjusted"],
                mode='markers',
                text=self.fraud_transactions["cc_num"],
                customdata=self.fraud_transactions[["amt", "city", "job", "first", "last"]],
                hovertemplate=(
                    "Cardholder: %{customdata[3]} %{customdata[4]}<br>"
                    "Amount: $%{customdata[0]}<br>"
                    "City: %{customdata[1]}<br>"
                    "Job: %{customdata[2]}"
                ),
                marker=dict(size=10, color='red', opacity=0.7),
                name="Cardholders"
            ))

        if show_merchants:
            # Merchants layer (unchanged)
            fig.add_trace(go.Scattermapbox(
                lat=self.fraud_transactions["merch_lat"],
                lon=self.fraud_transactions["merch_long"],
                mode='markers',
                text=self.fraud_transactions["merchant"],
                customdata=self.fraud_transactions[["category", "city"]],
                hovertemplate=(
                    "Merchant: %{text}<br>"
                    "Category: %{customdata[0]}<br>"
                    "City: %{customdata[1]}"
                ),
                marker=dict(size=10, color='blue', opacity=0.5),
                name="Merchants"
            ))

        fig.update_layout(
            mapbox_style="open-street-map",
            mapbox=dict(
                zoom=3,
                center={"lat": 37.7749, "lon": -122.4194}  # Center to San Francisco
            ),
            margin={"r": 0, "t": 0, "l": 0, "b": 0},
            height=600
        )

        return fig

    def setup_layout(self):
        self.app.layout = html.Div([
            dcc.Checklist(
                id='toggle-checklist',
                options=[
                    {'label': 'Show Cardholders', 'value': 'cardholders'},
                    {'label': 'Show Merchants', 'value': 'merchants'}
                ],
                value=['cardholders', 'merchants'],
                labelStyle={'display': 'block'}
            ),
            dcc.Graph(id='fraud-map', figure=self.fig),
            html.Div(id='click-data', children="Click on a point to see details", style={'margin-top': '20px'})
        ])

    def setup_callbacks(self):
        @self.app.callback(
            Output('fraud-map', 'figure'),
            Input('toggle-checklist', 'value')
        )
        def update_map(selected_values):
            show_cardholders = 'cardholders' in selected_values
            show_merchants = 'merchants' in selected_values
            return self.create_map_figure(show_cardholders, show_merchants)

        @self.app.callback(
            Output('click-data', 'children'),
            Input('fraud-map', 'clickData')
        )
        def display_click_data(clickData):
            if clickData is None:
                return "Click on a point to see details"
            
            point_data = clickData['points'][0]
            text = point_data.get('text', '')
            customdata = point_data.get('customdata', [])

            if text.isdigit():  # It's a cardholder
                return html.Div([
                    html.P(f"Card Holder: {customdata[3]} {customdata[4]}"),
                    html.P(f"Transaction Amount: ${customdata[0]}"),
                    html.P(f"City: {customdata[1]}"),
                    html.P(f"Job: {customdata[2]}")
                ])
            else:  # It's a merchant
                return html.Div([
                    html.P(f"Merchant: {text}"),
                    html.P(f"Category: {customdata[0]}"),
                    html.P(f"City: {customdata[1]}")
                ])

    def run(self, port=8050):
        self.app.run_server(debug=True, port=port)

