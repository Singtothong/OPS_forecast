import streamlit as st
import pandas as pd
import plotly.express as px
from fbprophet import Prophet
from fbprophet.plot import plot_plotly
from plotly import graph_objs as go

df_raw = pd.read_excel('BRD2021.xls')
df = pd.read_csv('exampledata.csv')

material_list = df['Material'].value_counts().index.to_list()

st.title('OPS Forecasting')
# Create sidebar
st.sidebar.header('Input Material')

def userInputFeatures():
    material_select = st.sidebar.selectbox('Material No.', material_list)
    n_month = st.sidebar.slider('Months of Prediction:', 1, 6)
    features = df[df['Material'] == material_select]
    raw_select = df_raw[df_raw['Material'] == material_select]
    return features, n_month, raw_select

inputData, n_month, raw_select = userInputFeatures()
period = n_month * 30
df_train = raw_select[['PR Date','Quantity']]
df_train = df_train.rename(columns={"PR Date": "ds", "Quantity": "y"})

m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

fig = px.bar(inputData, x="PR_Month", y=["PR_Qty", "Delivery_Qty"], barmode='group',labels={"variable": "Category"}, height=400)
fig.update_layout(
    title="Damand by month",
    xaxis_title="Month",
    yaxis_title="Quantity",
)
st.plotly_chart(fig)

# Show and plot forecast
st.subheader('Forecast data')
st.write(forecast.tail())

st.write(f'Forecast plot for {n_month} months')
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)
