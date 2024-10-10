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
