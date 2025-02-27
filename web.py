import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from joblib import dump, load
import plotly.graph_objects as go

@st.cache_data(ttl=3600)
def load_data(filename='Data_total.csv'):
    data = pd.read_csv(filename)
    data = data.drop('Month', axis=1, errors='ignore')
    return data

def clean_data(data):
    columns = ['Inbound_Inter', 'Outbound_Inter', 'Inbound_Dom', 'Outbound_Dom', 'Total In', 'Total Out']
    for column in columns:
        if data[column].dtype == 'object':
            data[column] = data[column].str.replace(',', '').astype(float)
    return data


def train_arima_model(series, order=(1, 1, 1)):
    model = ARIMA(series, order=order)
    model_fit = model.fit()
    return model_fit


def forecast_arima_model(model_fit, steps=1):
    forecast = model_fit.forecast(steps=steps)
    return forecast


def save_model(model, variable):
    filename = f'{variable}_arima_model.pkl'
    with open(filename, 'wb') as pkl:
        dump(model, pkl)


def load_model(variable):
    filename = f'{variable}_arima_model.pkl'
    with open(filename, 'rb') as pkl:
        model = load(pkl)
    return model


def main():
    st.set_page_config(layout="wide")
    data_file = 'Data_total.csv'
    data = load_data(data_file)
    data = clean_data(data)
    option = st.sidebar.selectbox("Choose an option:", ["Add New Data", "Predictions"])

    if option == "Add New Data":
        st.subheader("Add New Data Entry")
        with st.form("new_data_form"):
            inbound_inter = st.number_input('Inbound International', min_value=0.00, format="%.2f")
            outbound_inter = st.number_input('Outbound International', min_value=0.00, format="%.2f")
            inbound_dom = st.number_input('Inbound Domestic', min_value=0.00, format="%.2f")
            outbound_dom = st.number_input('Outbound Domestic', min_value=0.00, format="%.2f")
            submit_button = st.form_submit_button("Submit and Retrain")

            # Check if all fields are filled
            if submit_button:
                if any([v == 0.00 for v in [inbound_inter, outbound_inter, inbound_dom, outbound_dom]]):
                    st.warning("Please enter all fields before submitting.")
                else:
                    # Proceed with retraining or data processing here
                    new_data = pd.DataFrame({
                        'Inbound_Inter': [inbound_inter],
                        'Outbound_Inter': [outbound_inter],
                        'Inbound_Dom': [inbound_dom],
                        'Outbound_Dom': [outbound_dom]
                    })
                    updated_data = pd.concat([data, new_data], ignore_index=True)
                    updated_data.to_csv(data_file, index=False)

                for var in ['Inbound_Inter', 'Outbound_Inter', 'Inbound_Dom', 'Outbound_Dom']:
                    series = updated_data[var].dropna()
                    model_fit = train_arima_model(series)
                    save_model(model_fit, var)

                st.success("Data added and models retrained.")

    elif option == "Predictions":
        forecasts = {}
        for var in ['Inbound_Inter', 'Outbound_Inter', 'Inbound_Dom', 'Outbound_Dom']:
            model_fit = load_model(var)
            forecast = forecast_arima_model(model_fit, steps=1)
            forecasts[var] = forecast.iloc[0]

        fig = go.Figure(data=[
            go.Bar(
                x=list(forecasts.keys()),
                y=list(forecasts.values()),
                marker_color=['#FF6384', '#36A2EB', '#FFCE56', '#4BC0C0'],
                hoverinfo='y',
                hovertemplate='%{y:.2f}<extra></extra>'
            )
        ])
        fig.update_layout(
            title={
                'text': "Predicted Tonnage Next Month",  # Tiêu đề của biểu đồ
                'y': 1,  # Vị trí trục y của tiêu đề, giá trị từ 0 đến 1
                'x': 0.5,  # Vị trí trục x của tiêu đề, căn giữa là 0.5
                'xanchor': 'center',  # Căn tiêu đề tại vị trí x
                'yanchor': 'top'  # Căn tiêu đề tại vị trí y
            },
            title_font=dict(size=32, color='blue'),  # Thiết lập font chữ và màu sắc cho tiêu đề
            xaxis_title="Category",
            yaxis_title="Volume (Ton)",
            autosize=False,
            width=600,  # Chiều rộng của biểu đồ
            height=800,  # Chiều cao của biểu đồ
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
