import streamlit as st
import folium
import pandas as pd
from streamlit_folium import st_folium

# Load airport data (assuming a CSV file is available with latitude, longitude, continent)
# You might need to adjust the path or structure of the data as per your setup
url = 'https://raw.githubusercontent.com/jpatokal/openflights/master/data/airports.dat'
columns = ["Airport ID", "Name", "City", "Country", "IATA", "ICAO", "Latitude", "Longitude", "Altitude", 
           "Timezone", "DST", "Tz database time zone", "Type", "Source"]
airports = pd.read_csv(url, header=None, names=columns)

# Hard-coded mapping of continents to countries (simplified list)
continent_country_map = {
    'Africa': ['Algeria', 'Nigeria', 'South Africa', 'Egypt', 'Kenya', 'Morocco', 'Ethiopia'],
    'Asia': ['China', 'India', 'Japan', 'South Korea', 'Thailand', 'Indonesia', 'Saudi Arabia'],
    'Europe': ['France', 'Germany', 'United Kingdom', 'Italy', 'Spain', 'Poland', 'Sweden'],
    'North America': ['United States', 'Canada', 'Mexico', 'Cuba', 'Panama', 'Honduras'],
    'South America': ['Brazil', 'Argentina', 'Colombia', 'Chile', 'Peru', 'Venezuela'],
    'Oceania': ['Australia', 'New Zealand', 'Fiji', 'Papua New Guinea', 'Samoa']
}

# Function to get countries by continent
def get_countries_by_continent(continent):
    return continent_country_map.get(continent, [])

# Streamlit dropdown for continent selection
continents = ['Africa', 'Asia', 'Europe', 'North America', 'South America', 'Oceania']
selected_continent = st.selectbox("Select a Continent", continents)

# Filter the airports based on the selected continent
continent_airports = airports[airports['Country'].isin(get_countries_by_continent(selected_continent))]

# Initialize Folium map centered on the selected continent's coordinates
continent_centers = {
    "Africa": [0, 20],
    "Asia": [34, 100],
    "Europe": [54, 15],
    "North America": [40, -100],
    "South America": [-15, -60],
    "Oceania": [-25, 140]
}

# Create Folium map centered on the continent and set to fixed zoom level
map_center = continent_centers[selected_continent]
m = folium.Map(location=map_center, zoom_start=3, max_bounds=True, dragging=False, scrollWheelZoom=False)

# Add circle markers (dots) for the airports
for _, airport in continent_airports.iterrows():
    folium.CircleMarker(
        location=[airport['Latitude'], airport['Longitude']],
        radius=5,  # Adjust radius to control the size of the dot
        color='red',  # You can customize the color
        fill=True,
        fill_color='blue',
        fill_opacity=0.6,
        popup=f"{airport['Name']} ({airport['IATA']})"
    ).add_to(m)

# Display the map in Streamlit
st_folium(m, width=700, height=500)
