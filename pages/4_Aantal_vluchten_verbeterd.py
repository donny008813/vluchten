import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the data
csv_url = 'https://raw.githubusercontent.com/donny008813/vluchten/main/schedule_airport.csv'
df = pd.read_csv(csv_url, sep=',', on_bad_lines='skip')

# Dictionary of airlines
airline_codes = {
    "A3": "Aegean Airlines", "AA": "American Airlines", "AC": "Air Canada", "AF": "Air France",
    "AT": "Royal Air Maroc", "AY": "Finnair", "AZ": "Alitalia", "BA": "British Airways",
    "BT": "Air Baltic", "CJ": "CityJet", "CX": "Cathay Pacific", "DL": "Delta Air Lines",
    "EI": "Aer Lingus", "EK": "Emirates", "EW": "Eurowings", "EY": "Etihad Airways",
    "EZY": "easyJet", "FB": "Bulgaria Air", "FI": "Icelandair", "GM": "Germania",
    "IB": "Iberia", "JP": "Adria Airways", "JU": "Air Serbia", "KL": "KLM",
    "LH": "Lufthansa", "LO": "LOT Polish Airlines", "LX": "Swiss International Air Lines",
    "LY": "El Al", "OS": "Austrian Airlines", "OU": "Croatia Airlines", "PC": "Pegasus Airlines",
    "PS": "Ukraine International Airlines", "QR": "Qatar Airways", "RJ": "Royal Jordanian",
    "SK": "SAS", "SQ": "Singapore Airlines", "SU": "Aeroflot", "TG": "Thai Airways",
    "TK": "Turkish Airlines", "TP": "TAP Air Portugal", "U6": "Ural Airlines",
    "UA": "United Airlines", "UX": "Air Europa", "VY": "Vueling", "WK": "Edelweiss Air",
    "WY": "Oman Air", "XQ": "SunExpress", "YM": "Montenegro Airlines"
}

# Function to identify airline
def identify_airline(flight_number):
    airline_code = flight_number[:2]
    return airline_codes.get(airline_code, 'Unknown Airline')

# Prepare the data
df['Airline_Full_Name'] = df['FLT'].apply(identify_airline)
df['Date'] = pd.to_datetime(df['STD'], format='%d/%m/%Y')
df['Time'] = pd.to_datetime(df['STA_STD_ltc'], format='%H:%M:%S').dt.time
df['DateTime'] = pd.to_datetime(df['Date'].astype(str) + " " + df['Time'].astype(str))
df['Hour'] = pd.to_datetime(df['STA_STD_ltc'], format='%H:%M:%S').dt.hour

# Unique airlines
unique_airlines = df['Airline_Full_Name'].unique()

# Subsets for departing and arriving flights
departing_flights = df[df['LSV'] == 'S']
arriving_flights = df[df['LSV'] == 'L']

# Variables for plots
departing_flights_per_day = departing_flights.groupby('Date').size()
arriving_flights_per_day = arriving_flights.groupby('Date').size()

departing_flights_per_month = departing_flights.groupby([departing_flights['Date'].dt.year, departing_flights['Date'].dt.month]).size()
arriving_flights_per_month = arriving_flights.groupby([arriving_flights['Date'].dt.year, arriving_flights['Date'].dt.month]).size()

departing_flights_per_airline = departing_flights.groupby('Airline_Full_Name').size()
arriving_flights_per_airline = arriving_flights.groupby('Airline_Full_Name').size()

# Streamlit App
st.markdown("# Vertrekkende en landende vluchten")
st.sidebar.header("Vertrekkende en landende vluchten")
st.write(
    """Op deze pagina staan gegevens over het opstijgen en landen van de vluchten gedurende de periode van de dataset. Ook kunnen de verschillen in vluchten tussen de verschillende luchtvaartmaatschapij gemakkelijk vergeleken worden."""
)

# Dropdown menu to select arriving or departing flights
flight_type = st.selectbox('Selecteer landende of vertrekkende vluchten', ['Vertrekkende vluchten', 'Landende vluchten', 'Beide'])

# Plotting the selected flight type (daily chart)
if flight_type == 'Vertrekkende vluchten':
    st.subheader('Vertrekkende vluchten per dag')
    st.line_chart(departing_flights_per_day)
elif flight_type == 'Landende vluchten':
    st.subheader('Landende vluchten per dag')
    st.line_chart(arriving_flights_per_day)
else:
    st.subheader('Vertrekkende en landende vluchten per dag')
    combined_flights = departing_flights_per_day + arriving_flights_per_day
    st.line_chart(combined_flights)

# Airline toggle for the monthly chart
st.header('Aantal vluchten per maatschapij')
selected_airlines = st.multiselect('Selecteer maatschapij', unique_airlines, default=unique_airlines)

# Filter data for selected airlines
# Ensure that only valid airlines that are present in the dataset are selected
valid_airlines = [airline for airline in selected_airlines if airline in departing_flights_per_airline.index]

# Check if the valid airlines list is not empty
if valid_airlines:
    # Filter the dataframes using the valid airlines
    filtered_departing_flights_per_airline = departing_flights_per_airline[departing_flights_per_airline.index.isin(valid_airlines)]
    filtered_arriving_flights_per_airline = arriving_flights_per_airline[arriving_flights_per_airline.index.isin(valid_airlines)]

    # Combine the selected airlines data
    flight_type_airline = st.selectbox('Selecteer type vlucht', ['Vertrekkende vluchten', 'Landende vluchten', 'Beide'])

    if flight_type_airline == 'Vertrekkende vluchten':
        st.subheader('Vertrekkende vluchten per maatschapij')
        st.bar_chart(filtered_departing_flights_per_airline)
    elif flight_type_airline == 'Landende vluchten':
        st.subheader('Landende vluchten per maatschapij')
        st.bar_chart(filtered_arriving_flights_per_airline)
    else:
        st.subheader('Landende en vertrekkende vluchten per maatschapij')
        combined_airline_flights = filtered_departing_flights_per_airline + filtered_arriving_flights_per_airline
        st.bar_chart(combined_airline_flights)
else:
    st.warning("Geen selectie gemaakt, selecteer minimaal 1 maatschapij.")
