import streamlit as st
import time
import numpy as np
import pandas as pd
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta


st.set_page_config(page_title="Vergelijking van opstijgen en landen")

st.markdown("# Opstijgen en landen")
st.sidebar.header("Opstijgen en landen")
st.write(
    """In deze grafiek zullen van verschillende vluchten het opsteigen en landen getoond worden.
    Hierin is voor elke vlucht de hoogte en de tijd te zien."""
)

flight_1_30 = pd.read_excel('https://raw.githubusercontent.com/donny008813/vluchten/main/30Flight1.xlsx')
flight_2_30 = pd.read_excel('https://raw.githubusercontent.com/donny008813/vluchten/main/30Flight2.xlsx')
flight_3_30 = pd.read_excel('https://raw.githubusercontent.com/donny008813/vluchten/main/30Flight3.xlsx')
flight_4_30 = pd.read_excel('https://raw.githubusercontent.com/donny008813/vluchten/main/30Flight4.xlsx')
flight_5_30 = pd.read_excel('https://raw.githubusercontent.com/donny008813/vluchten/main/30Flight5.xlsx')
flight_6_30 = pd.read_excel('https://raw.githubusercontent.com/donny008813/vluchten/main/30Flight6.xlsx')
flight_7_30 = pd.read_excel('https://raw.githubusercontent.com/donny008813/vluchten/main/30Flight7.xlsx')

flight_1_30 = flight_1_30[flight_1_30['[3d Altitude Ft]'] >= 500]
y1_30 = flight_1_30['[3d Altitude Ft]']
flight_2_30 = flight_2_30[flight_2_30['[3d Altitude Ft]'] >= 500]
y2_30 = flight_2_30['[3d Altitude Ft]']
flight_3_30 = flight_3_30[flight_3_30['[3d Altitude Ft]'] >= 500]
y3_30 = flight_3_30['[3d Altitude Ft]']
flight_4_30 = flight_4_30[flight_4_30['[3d Altitude Ft]'] >= 500]
y4_30 = flight_4_30['[3d Altitude Ft]']
flight_5_30 = flight_5_30[flight_5_30['[3d Altitude Ft]'] >= 500]
y5_30 = flight_5_30['[3d Altitude Ft]']
flight_6_30 = flight_6_30[flight_6_30['[3d Altitude Ft]'] >= 500]
y6_30 = flight_6_30['[3d Altitude Ft]']
flight_7_30 = flight_7_30[flight_7_30['[3d Altitude Ft]'] >= 500]
y7_30 = flight_7_30['[3d Altitude Ft]']

x_1tijd = flight_1_30['Time (secs)']
x_2tijd = flight_2_30['Time (secs)']
x_3tijd = flight_3_30['Time (secs)']
x_4tijd = flight_4_30['Time (secs)']
x_5tijd = flight_5_30['Time (secs)']
x_6tijd = flight_6_30['Time (secs)']
x_7tijd = flight_7_30['Time (secs)']

fig1, ax1 = plt.subplots()
sns.lineplot(data=flight_1_30, x=x_1tijd, y=y1_30, label='vlucht 1')
sns.lineplot(data=flight_2_30, x=x_2tijd, y=y2_30, label='vlucht 2')
sns.lineplot(data=flight_3_30, x=x_3tijd, y=y3_30, label='vlucht 3')
sns.lineplot(data=flight_4_30, x=x_4tijd, y=y4_30, label='vlucht 4')
sns.lineplot(data=flight_5_30, x=x_5tijd, y=y5_30, label='vlucht 5')
sns.lineplot(data=flight_6_30, x=x_6tijd, y=y6_30, label='vlucht 6')
sns.lineplot(data=flight_7_30, x=x_7tijd, y=y7_30, label='vlucht 7')
plt.legend()
ax1.set_xlabel('Tijd in seconde')
ax1.set_ylabel('Altitude in feets')
ax1.set_title('Altitude tegenover tijd in seconde van de 7 vluchten')
st.pyplot(fig1)
