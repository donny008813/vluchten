import streamlit as st
import folium
import pandas as pd
from streamlit_folium import st_folium

# Set page configuration
st.set_page_config(page_title="Hallo en welkom bij de vliegveldenmap")

# Title and description
st.write("# Wereldwijd gebruikte vliegvelden voor landen en opstijgen")
st.markdown("""
    In deze interactieve map kun je de drukte op verschillende vliegvelden inzien. Hierbij kun je filteren op landende of vertrekkende vluchten, de periode waarin deze vluchten hebben plaatsgevonden en filteren op drukte van de vliegvelden.
""")

# Load airport data
url_csv = 'https://raw.githubusercontent.com/donny008813/vluchten/main/airports-extended-clean.csv'
df = pd.read_csv(url_csv, sep=';', decimal=',')

# Load flight data
csv_url = 'https://raw.githubusercontent.com/donny008813/vluchten/main/schedule_airport.csv'
vluchten = pd.read_csv(csv_url)

# Convert the 'STD' column to datetime
vluchten['STD'] = pd.to_datetime(vluchten['STD'], errors='coerce')

# Create columns for map and checkboxes
col1, col2 = st.columns([3, 1])  # Make the left column wider than the right column

# Place dropdown and date slider
with col1:
    # Dropdown for selecting departing or arriving flights
    flight_type = st.selectbox("Vlucht type:", ["Vertrekkend", "Aankomend"])

    # Slider for selecting the date range
    min_date = vluchten['STD'].min().to_pydatetime()
    max_date = vluchten['STD'].max().to_pydatetime()

    start_date, end_date = st.slider(
        "Select Date Range:",
        min_value=min_date,
        max_value=max_date,
        value=(min_date, max_date),
        format="YYYY-MM-DD"
    )

# Filter flight data based on the selected flight type and date range
if flight_type == "Vertrekkend":
    # Filter for departing flights
    filtered_vluchten = vluchten[(vluchten['LSV'] == 'S') & (vluchten['STD'] >= start_date) & (vluchten['STD'] <= end_date)]
else:
    # Filter for arriving flights
    filtered_vluchten = vluchten[(vluchten['LSV'] == 'L') & (vluchten['STD'] >= start_date) & (vluchten['STD'] <= end_date)]

# Recalculate flight counts based on the selected flight type and date range
flight_counts = filtered_vluchten['Org/Des'].value_counts().reset_index()
flight_counts.columns = ['ICAO', 'Flight Count']

# Merge the flight counts with airport coordinates and names
vliegvelden_comp = df[['ICAO', 'Longitude', 'Latitude', 'Name']]  # Include airport names
locaties = vliegvelden_comp.merge(flight_counts, on='ICAO', how='inner')

# Calculate the 10th percentiles for dynamic coloring and checkbox ranges
percentiles = locaties['Flight Count'].quantile([i/10 for i in range(1, 11)]).to_dict()

# Dynamically set checkbox ranges based on the percentiles
with col2:
    st.write("### Vliegvelden per vluchten:")
    colors = ['#00FF00', '#00CC00', '#009900', '#336600', '#669900', '#996600', '#CC6600', '#FF6600', '#FF3300', '#FF0000']
    checkboxes = []
    
    for i in range(10):
        lower_bound = percentiles.get(i/10, 0)
        upper_bound = percentiles.get((i+1)/10, float('inf'))
        label = f"{lower_bound:,.0f} - {upper_bound:,.0f} vluchten"
        checkbox = st.checkbox(label, value=True)
        checkboxes.append((checkbox, lower_bound, upper_bound, colors[i]))

# Filter the DataFrame based on the selected checkboxes using apply
def filter_airports(row):
    # Check if the flight count falls within the selected range for each checkbox
    for checkbox, lower_bound, upper_bound, _ in checkboxes:
        if lower_bound <= row['Flight Count'] < upper_bound and checkbox:
            return True
    return False

filtered_locaties = locaties[locaties.apply(filter_airports, axis=1)]

# Function to determine the color based on the percentiles
def get_color(flight_count):
    for checkbox, lower_bound, upper_bound, color in checkboxes:
        if lower_bound <= flight_count < upper_bound and checkbox:
            return color
    return '#FFFFFF'  # Default color

# Normalize the flight count to scale the radius
def get_dynamic_radius(flight_count):
    max_flights = filtered_locaties['Flight Count'].max()
    min_radius = 5  # Minimum radius size
    max_radius = 30  # Maximum radius size

    # Scale radius to be between min_radius and max_radius
    radius = min_radius + (flight_count / max_flights) * (max_radius - min_radius)
    return radius

# Initialize Folium map with a global view
m = folium.Map(location=[0, 0], zoom_start=2)

# Adding markers for each airport with dynamic radius and color
for idx, row in filtered_locaties.iterrows():
    radius = get_dynamic_radius(row['Flight Count'])

    folium.CircleMarker(
        location=[row['Latitude'], row['Longitude']],
        radius=radius,
        color=get_color(row['Flight Count']),
        fill=True,
        fill_color=get_color(row['Flight Count']),
        popup=f"{row['Name']} - {row['Flight Count']} vluchten"  # Display airport name
    ).add_to(m)

# Display the map in the left column
with col1:
    st.write("### Luchthavens kaart")
    st_folium(m, width=700, height=500)

# Further Details Section Below the Map
st.write("### Overige informatie")
st.markdown("""
    Hier kun je meer gedetaileerde informatie vinden van de top 5 drukste en top 5 rustigste vliegvelden, gebasseerd op de periode eerder geselecteerd.
""")

# Display top 5 and bottom 5 airports based on the number of flights during the selected period
top_5_airports = filtered_vluchten['Org/Des'].value_counts().head(5).reset_index()
top_5_airports.columns = ['ICAO', 'Flight Count']
top_5_airports = top_5_airports.merge(df[['ICAO', 'Name']], on='ICAO', how='left')

bottom_5_airports = filtered_vluchten['Org/Des'].value_counts().tail(5).reset_index()
bottom_5_airports.columns = ['ICAO', 'Flight Count']
bottom_5_airports = bottom_5_airports.merge(df[['ICAO', 'Name']], on='ICAO', how='left')

# Display top 5 most used airports
st.write("### Top 5 meest gebruikte vliegvelden")
for index, row in top_5_airports.iterrows():
    st.write(f"**{row['Name']}** ({row['ICAO']}) - {row['Flight Count']} vluchten")

# Display bottom 5 least used airports
st.write("### Top 5 minst gebruikte vliegvelden")
for index, row in bottom_5_airports.iterrows():
    st.write(f"**{row['Name']}** ({row['ICAO']}) - {row['Flight Count']} vluchten")
