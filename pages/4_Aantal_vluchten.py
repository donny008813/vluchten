# VERSIE 1
 
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
 
# Load the data
csv_url = 'https://raw.githubusercontent.com/donny008813/vluchten/main/schedule_airport.csv'
df = pd.read_csv(csv_url, sep=',', on_bad_lines='skip')
 
# Dictionary of airlines and airports
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
 
airport_dict = {
    "LGAV": "Athens International Airport", "KPHL": "Philadelphia International Airport",
    "LFPG": "Charles de Gaulle Airport", "GMMN": "Mohammed V International Airport",
    "EFHK": "Helsinki-Vantaa Airport", "LIRF": "Leonardo da Vinci–Fiumicino Airport",
    "EGLL": "Heathrow Airport", "EVRA": "Riga International Airport",
    "EGLC": "London City Airport", "VHHH": "Hong Kong International Airport",
    "KJFK": "John F. Kennedy International Airport", "EIDW": "Dublin Airport",
    "OMDB": "Dubai International Airport", "EDDK": "Cologne Bonn Airport",
    "EDDH": "Hamburg Airport", "EDDL": "Düsseldorf Airport",
    "OMAA": "Abu Dhabi International Airport", "EGGW": "London Luton Airport",
    "LIRN": "Naples International Airport", "EDDT": "Berlin Tegel Airport",
    "LPPR": "Porto Airport", "LEMD": "Adolfo Suárez Madrid–Barajas Airport",
    "LEBL": "Barcelona–El Prat Airport", "LWSK": "Skopje International Airport",
    "BKPR": "Pristina International Airport", "LJLJ": "Ljubljana Jože Pučnik Airport",
    "LYBE": "Belgrade Nikola Tesla Airport", "EHAM": "Amsterdam Airport Schiphol",
    "EDDF": "Frankfurt Airport", "EPWA": "Warsaw Chopin Airport",
    "KEWR": "Newark Liberty International Airport", "KBOS": "Logan International Airport",
    "KMIA": "Miami International Airport", "VIDP": "Indira Gandhi International Airport",
    "VABB": "Chhatrapati Shivaji Maharaj International Airport", "RJAA": "Narita International Airport",
    "WSSS": "Singapore Changi Airport", "VTBS": "Suvarnabhumi Airport",
    "ZSPD": "Shanghai Pudong International Airport", "ZBAA": "Beijing Capital International Airport",
    "HECA": "Cairo International Airport", "OOMS": "Muscat International Airport",
    "LLBG": "Ben Gurion Airport", "FAJS": "O. R. Tambo International Airport",
    "SAEZ": "Ministro Pistarini International Airport", "SBGL": "Rio de Janeiro/Galeão International Airport",
    "HTDA": "Julius Nyerere International Airport", "EGCC": "Manchester Airport",
    "EGBB": "Birmingham Airport", "LFBD": "Bordeaux–Mérignac Airport",
    "LFMN": "Nice Côte d'Azur Airport", "ESSA": "Stockholm Arlanda Airport",
    "EKCH": "Copenhagen Airport", "ULLI": "Pulkovo Airport", "UUDD": "Domodedovo International Airport",
    "LKPR": "Václav Havel Airport Prague", "LOWW": "Vienna International Airport",
    "LIMC": "Milan Malpensa Airport", "LIPZ": "Venice Marco Polo Airport",
    "LIRQ": "Florence Airport", "LIRN": "Naples International Airport",
    "LROP": "Henri Coandă International Airport", "LSGG": "Geneva Airport",
    "LSZA": "Lugano Airport", "LBSF": "Sofia Airport", "BIKF": "Keflavík International Airport",
    "GCLA": "La Palma Airport", "OLBA": "Beirut–Rafic Hariri International Airport",
    "HEMA": "Marsa Alam International Airport", "LWOH": "Lviv Danylo Halytskyi International Airport"
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
 
# Unique airlines and airports
unique_airlines = df['Airline_Full_Name'].unique()
unique_airports = df['Org/Des'].unique()
 
# Color palettes
airline_colors = sns.color_palette("hsv", len(unique_airlines))
airport_colors = sns.color_palette("hsv", len(unique_airports))
airline_color_dict = {airline: color for airline, color in zip(unique_airlines, airline_colors)}
airport_color_dict = {airport: color for airport, color in zip(unique_airports, airport_colors)}
 
# Subsets for departing and arriving flights
departing_flights = df[df['LSV'] == 'S']
arriving_flights = df[df['LSV'] == 'L']
 
# Variables for plots
departing_flights_per_day = departing_flights.groupby('Date').size()
arriving_flights_per_day = arriving_flights.groupby('Date').size()
departing_flights_per_day_per_airline = departing_flights.groupby(['Date', 'Airline_Full_Name']).size().unstack(fill_value=0)
arriving_flights_per_day_per_airline = arriving_flights.groupby(['Date', 'Airline_Full_Name']).size().unstack(fill_value=0)
most_used_arrival_airports = arriving_flights.groupby(['Airline_Full_Name', 'Org/Des']).size().unstack(fill_value=0)
 
# Streamlit app
st.title('Flight Schedule Analysis')
st.header('Overview of Flight Data')
 
st.subheader('Data Preview')
st.write(df.head())
 
st.subheader('Departing Flights Per Day')
st.line_chart(departing_flights_per_day)
 
st.subheader('Arriving Flights Per Day')
st.line_chart(arriving_flights_per_day)
 
st.subheader('Departing Flights Per Day Per Airline')
st.line_chart(departing_flights_per_day_per_airline)
 
st.subheader('Arriving Flights Per Day Per Airline')
st.line_chart(arriving_flights_per_day_per_airline)
 
st.subheader('Most Used Arrival Airports Per Airline')
st.write(most_used_arrival_airports)
