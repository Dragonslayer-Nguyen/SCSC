import streamlit as st
import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go

# Đọc dữ liệu
data = pd.read_csv('Data_total.csv')
data = data.drop('Month', axis=1, errors='ignore')

# Các hàm đã được định nghĩa ở trên
def clean_data(data):
    columns = ['Inbound_Inter', 'Outbound_Inter', 'Inbound_Dom', 'Outbound_Dom', 'Total In', 'Total Out']
    for column in columns:
        if data[column].dtype == 'object':
            data[column] = data[column].str.replace(',', '').astype(float)
    return data

def prepare_data(data, time_steps):
    X, y = [], []
    feature_columns = data.columns.tolist()
    for i in range(time_steps, len(data)):
        X.append(data.iloc[i - time_steps:i][feature_columns].values)
        y.append(data.iloc[i][['Inbound_Inter', 'Outbound_Inter', 'Inbound_Dom', 'Outbound_Dom']].values)
    return np.array(X), np.array(y)

def create_lstm_model(input_shape, output_units):
    model = Sequential([
        LSTM(100, return_sequences=True, input_shape=input_shape),
        LSTM(100),
        Dense(output_units)
    ])
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model
def model_LSTM(data, time_steps=12, epochs=50, batch_size=12):
    data = clean_data(data)
    data_features = data.values
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_features)

    X, y = prepare_data(pd.DataFrame(data_scaled, columns=data.columns), time_steps)

    model = create_lstm_model((time_steps, data.shape[1]), 4)  # Output units = 4 for four columns
    model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=1)
    return model, scaler

def output_LSTM(model, scaler, data, time_steps=12):
    data = clean_data(data)
    data_features = data.values
    data_scaled = scaler.transform(data_features)

    X = np.array([data_scaled[-time_steps:]])
    predicted_values = model.predict(X)
    predicted_values = scaler.inverse_transform(
        np.concatenate([predicted_values, np.zeros((predicted_values.shape[0], data.shape[1] - 4))], axis=1))[:, :4]

    return predicted_values

# # Đặt tiêu đề trên Streamlit
# st.title('Predicted Tonnage Visualization')

# Chạy mô hình và lấy output
model, scaler = model_LSTM(data, time_steps=12, epochs=50, batch_size=12)
output = output_LSTM(model, scaler, data, time_steps=12)
output_values = output.flatten()

st.set_page_config(layout="wide")
# Các nhãn tương ứng cho mỗi giá trị
labels = ['Inbound_Inter', 'Outbound_Inter', 'Inbound_Dom', 'Outbound_Dom']

# Tạo biểu đồ cột
fig = go.Figure(data=[
    go.Bar(
        x=labels,
        y=output_values,
        marker_color=['#FF6384', '#36A2EB', '#FFCE56', '#4BC0C0'],
        hoverinfo='y',
        hovertemplate='%{y:.2f}<extra></extra>'
    )
])

fig.update_layout(
    title={
        'text': "Predicted Tonnage",  # Tiêu đề của biểu đồ
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


# Hiển thị biểu đồ trên Streamlit, sử dụng toàn bộ chiều rộng của container
st.plotly_chart(fig, use_container_width=True)
