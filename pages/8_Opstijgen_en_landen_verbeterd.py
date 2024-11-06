import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Vergelijking van opstijgen en landen")

st.markdown("# Opstijgen en landen")
st.sidebar.header("Opstijgen en landen")
st.write(
    """In deze grafiek zullen van verschillende vluchten het opstijgen en landen getoond worden.
    Hierin is voor elke vlucht de hoogte en de tijd te zien."""
)

# Load the data
flight_1_30 = pd.read_excel('https://raw.githubusercontent.com/donny008813/vluchten/main/30Flight1.xlsx')
flight_2_30 = pd.read_excel('https://raw.githubusercontent.com/donny008813/vluchten/main/30Flight2.xlsx')
flight_3_30 = pd.read_excel('https://raw.githubusercontent.com/donny008813/vluchten/main/30Flight3.xlsx')
flight_4_30 = pd.read_excel('https://raw.githubusercontent.com/donny008813/vluchten/main/30Flight4.xlsx')
flight_5_30 = pd.read_excel('https://raw.githubusercontent.com/donny008813/vluchten/main/30Flight5.xlsx')
flight_6_30 = pd.read_excel('https://raw.githubusercontent.com/donny008813/vluchten/main/30Flight6.xlsx')
flight_7_30 = pd.read_excel('https://raw.githubusercontent.com/donny008813/vluchten/main/30Flight7.xlsx')

# Filter data
flight_1_30 = flight_1_30[flight_1_30['[3d Altitude Ft]'] >= 500]
flight_2_30 = flight_2_30[flight_2_30['[3d Altitude Ft]'] >= 500]
flight_3_30 = flight_3_30[flight_3_30['[3d Altitude Ft]'] >= 500]
flight_4_30 = flight_4_30[flight_4_30['[3d Altitude Ft]'] >= 500]
flight_5_30 = flight_5_30[flight_5_30['[3d Altitude Ft]'] >= 500]
flight_6_30 = flight_6_30[flight_6_30['[3d Altitude Ft]'] >= 500]
flight_7_30 = flight_7_30[flight_7_30['[3d Altitude Ft]'] >= 500]

# Convert altitude from feet to kilometers (1 foot = 0.3048 meters, 1 kilometer = 1000 meters)
flight_1_30['Altitude (km)'] = flight_1_30['[3d Altitude Ft]'] * 0.3048 / 1000
flight_2_30['Altitude (km)'] = flight_2_30['[3d Altitude Ft]'] * 0.3048 / 1000
flight_3_30['Altitude (km)'] = flight_3_30['[3d Altitude Ft]'] * 0.3048 / 1000
flight_4_30['Altitude (km)'] = flight_4_30['[3d Altitude Ft]'] * 0.3048 / 1000
flight_5_30['Altitude (km)'] = flight_5_30['[3d Altitude Ft]'] * 0.3048 / 1000
flight_6_30['Altitude (km)'] = flight_6_30['[3d Altitude Ft]'] * 0.3048 / 1000
flight_7_30['Altitude (km)'] = flight_7_30['[3d Altitude Ft]'] * 0.3048 / 1000

# Convert time from seconds to minutes
flight_1_30['Time (mins)'] = flight_1_30['Time (secs)'] / 60
flight_2_30['Time (mins)'] = flight_2_30['Time (secs)'] / 60
flight_3_30['Time (mins)'] = flight_3_30['Time (secs)'] / 60
flight_4_30['Time (mins)'] = flight_4_30['Time (secs)'] / 60
flight_5_30['Time (mins)'] = flight_5_30['Time (secs)'] / 60
flight_6_30['Time (mins)'] = flight_6_30['Time (secs)'] / 60
flight_7_30['Time (mins)'] = flight_7_30['Time (secs)'] / 60

# Use Streamlit columns to arrange the plot and checkboxes side by side
col1, col2 = st.columns([3, 1])

# Plotting in the left column
with col1:
    fig1, ax1 = plt.subplots()

    # Plot flights based on the checkbox selections
    if col2.checkbox("Toon vlucht 1", value=True):
        sns.lineplot(data=flight_1_30, x='Time (mins)', y='Altitude (km)', label='vlucht 1')
    if col2.checkbox("Toon vlucht 2", value=True):
        sns.lineplot(data=flight_2_30, x='Time (mins)', y='Altitude (km)', label='vlucht 2')
    if col2.checkbox("Toon vlucht 3", value=True):
        sns.lineplot(data=flight_3_30, x='Time (mins)', y='Altitude (km)', label='vlucht 3')
    if col2.checkbox("Toon vlucht 4", value=True):
        sns.lineplot(data=flight_4_30, x='Time (mins)', y='Altitude (km)', label='vlucht 4')
    if col2.checkbox("Toon vlucht 5", value=True):
        sns.lineplot(data=flight_5_30, x='Time (mins)', y='Altitude (km)', label='vlucht 5')
    if col2.checkbox("Toon vlucht 6", value=True):
        sns.lineplot(data=flight_6_30, x='Time (mins)', y='Altitude (km)', label='vlucht 6')
    if col2.checkbox("Toon vlucht 7", value=True):
        sns.lineplot(data=flight_7_30, x='Time (mins)', y='Altitude (km)', label='vlucht 7')

    # Get the minimum altitude value across all data to set as the y-axis lower limit
    min_altitude = min(
        flight_1_30['Altitude (km)'].min(),
        flight_2_30['Altitude (km)'].min(),
        flight_3_30['Altitude (km)'].min(),
        flight_4_30['Altitude (km)'].min(),
        flight_5_30['Altitude (km)'].min(),
        flight_6_30['Altitude (km)'].min(),
        flight_7_30['Altitude (km)'].min()
    )

    # Set y-axis limits to start at the minimum altitude
    ax1.set_ylim(bottom=min_altitude)

    # Get the current y-tick values and draw horizontal dotted lines at those positions
    y_ticks = ax1.get_yticks()
    for y_tick in y_ticks:
        ax1.axhline(y=y_tick, color='gray', linestyle='--', linewidth=0.5)

    # Move the legend to the top right corner
    plt.legend(loc='upper right')

    # Customize the plot
    ax1.set_xlabel('Tijd in minuten')
    ax1.set_ylabel('Altitude in kilometers')
    ax1.set_title('Altitude tegenover tijd in minuten van de 7 vluchten')

    # Display the plot in Streamlit
    st.pyplot(fig1)
