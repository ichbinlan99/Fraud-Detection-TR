import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import pandas as pd
import plotly.graph_objects as go

class FraudMap:
    def __init__(self, fraud_transactions):
        self.fraud_transactions = fraud_transactions
        # Clean up the credit card numbers
        self.fraud_transactions['cc_num'] = self.fraud_transactions['cc_num'].astype(str).str.strip()
        self.app = dash.Dash(__name__)
        self.fig = self.create_map_figure(show_cardholders=True, show_merchants=True)
        self.setup_layout()
        self.setup_callbacks()

    def create_map_figure(self, show_cardholders, show_merchants):
        fig = go.Figure()

        if show_cardholders:
            # Cardholders layer with clickable points
            fig.add_trace(go.Scattermapbox(
                lat=self.fraud_transactions["lat"],
                lon=self.fraud_transactions["long"],
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
            # Merchants layer with clickable points
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
