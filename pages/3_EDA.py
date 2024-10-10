import pandas as pd
import numpy as np
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import streamlit as st

csv_url = 'https://raw.githubusercontent.com/donny008813/vluchten/main/schedule_airport.csv'
vluchten = pd.read_csv(csv_url)

vluchten_copy = vluchten.copy()

vluchten_copy['STA_STD_ltc'] = pd.to_datetime(vluchten_copy['STA_STD_ltc'], format='%H:%M:%S').dt.time
vluchten_copy['ATA_ATD_ltc'] = pd.to_datetime(vluchten_copy['ATA_ATD_ltc'], format='%H:%M:%S').dt.time

# Function to convert time to total seconds since midnight
def time_to_seconds(time_obj):
    return time_obj.hour * 3600 + time_obj.minute * 60 + time_obj.second

# Function to calculate the difference in total seconds, allowing for negative values
def time_diff_in_seconds(start_time, end_time):
    start_seconds = time_to_seconds(start_time)
    end_seconds = time_to_seconds(end_time)
    
    # Calculate the difference in seconds
    diff_seconds = end_seconds - start_seconds
    
    return diff_seconds


# Apply the time_diff_in_minutes function to each row
vluchten_copy['verschil'] = vluchten_copy.apply(lambda row: time_diff_in_seconds(row['STA_STD_ltc'], row['ATA_ATD_ltc']), axis=1)

def vertraagd(x) :
    if x > 0 :
        return 1
    else: 
        return 0

vluchten_copy['vertraagd'] = vluchten_copy['verschil'].apply(vertraagd)

vluchten_copy['STD'] = pd.to_datetime(vluchten_copy['STD'], format = '%d/%m/%Y')

vluchten_copy['dag'] = vluchten_copy['STD'].dt.day
vluchten_copy['maand'] = vluchten_copy['STD'].dt.month

vluchten_vertraagd = vluchten_copy[vluchten_copy['vertraagd'] == 1]

st.title('EDA')

#Hist vertraagd
fig3, ax3 = plt.subplots()
sns.countplot(data=vluchten_copy, x = 'vertraagd')
ax3.set_title('Count of Delays and Non-Delays')
ax3.set_ylabel('Count')
st.pyplot(fig3)

# Hist vertraagd dag
fig1, ax1 = plt.subplots()
# Plot de vertragingen per dag
plt.figure(figsize=(10, 6))
sns.countplot(data=vluchten_copy, x='dag', hue='vertraagd')
# Labels en titel
ax1.set_title('Count of Delays and Non-Delays per Day')
ax1.set_ylabel('Count')
ax1.set_xlabel('Day')
# Plot de eerste plot
st.pyplot(fig1)

# Hist vertraagd maand
fig2, ax2 = plt.subplots()
# Plot de vertragingen per maand
plt.figure(figsize=(10,6))
sns.countplot(data=vluchten_copy, x='maand', hue='vertraagd')
# Labels en titel
ax2.set_title('Count of Delays and Non-Delays per Month')
ax2.set_ylabel('Count')
ax2.set_xlabel('Maand')
# Plot
st.pyplot(fig2)
