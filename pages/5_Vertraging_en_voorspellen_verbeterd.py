import pandas as pd
import numpy as np
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import streamlit as st
import sklearn.linear_model as lm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.dates as mdates

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

# Maakt een functie aan om de vluchtnummers aan de luchtvaartmaatschapijen te kunnen koppelen
def identify_airline(flight_number):
    airline_code = flight_number[:2]
    return airline_code

vluchten_copy['maatschappij'] = vluchten_copy['FLT'].apply(identify_airline)

st.title('Vertraging van vluchten voorspellen')
st.write(vluchten_copy)

vertraging_drempel_minuten = st.slider('Selecteer de minimum vertragingstijd in minuten:', min_value=0, max_value=60, value=1)

# Pas de kolom 'vertraagd' aan op basis van de drempelwaarde in minuten
vertraging_drempel_seconden = vertraging_drempel_minuten * 60
vluchten_copy['vertraagd'] = vluchten_copy['verschil'].apply(lambda x: 1 if x > vertraging_drempel_seconden else 0)

# Toon de bijgewerkte grafiek
fig, ax = plt.subplots()
sns.countplot(data=vluchten_copy, x='vertraagd')
ax.set_title('Aantal Vertragingen en Geen-Vertragingen')
ax.set_ylabel('Aantal')
st.pyplot(fig)

#Hist vertraagd
fig1, ax1 = plt.subplots()
sns.countplot(data=vluchten_copy, x = 'vertraagd')
ax1.set_title('Count of Delays and Non-Delays')
ax1.set_ylabel('Count')
st.pyplot(fig1)

# Hist vertraagd dag
fig2, ax2 = plt.subplots()
# Plot de vertragingen per dag
sns.countplot(data=vluchten_copy, x='dag', hue='vertraagd')
# Labels en titel
ax2.set_title('Count of Delays and Non-Delays per Day')
ax2.set_ylabel('Count')
ax2.set_xlabel('Day')
# Plot de eerste plot
st.pyplot(fig2)

# Hist vertraagd maand
fig3, ax3 = plt.subplots()
# Plot de vertragingen per maand
sns.countplot(data=vluchten_copy, x='maand', hue='vertraagd')
# Labels en titel
ax3.set_title('Count of Delays and Non-Delays per Month')
ax3.set_ylabel('Count')
ax3.set_xlabel('Month')
# Plot
st.pyplot(fig3)

# Hist vertraagd maatschappij
flight_counts = vluchten_copy['maatschappij'].value_counts().nlargest(10).index

# Filtering DataFrame for top 10 airlines
top_10_df = vluchten_copy[vluchten_copy['maatschappij'].isin(flight_counts)]

fig4, ax4 = plt.subplots()
sns.countplot(data=top_10_df, x='maatschappij', hue='vertraagd')
ax4.set_title('Count of Delays and Non-Delays per Airline')
ax4.set_ylabel('Count')
ax4.set_xlabel('Airline')
st.pyplot(fig4)

# Group by date and count the number of delayed flights
delayed_counts = vluchten_copy.groupby('STD')['vertraagd'].sum().reset_index()

fig5, ax5 = plt.subplots()

sns.barplot(x='STD', y='vertraagd', data=delayed_counts, color='lightblue')

ax5.set_title('Number of Delayed Flights per Day')
ax5.set_xlabel('Date')
ax5.set_ylabel('Number of Delayed Flights')
ax5.tick_params(axis='x', rotation=45)
ax5.xaxis.set_major_locator(mdates.YearLocator())  # Set major ticks for each year
plt.tight_layout()

st.pyplot(fig5)

st.subheader('Voorspellen van vertraging')
model_data = vluchten_copy.copy()

airline = pd.get_dummies(model_data['maatschappij'])
maand = pd.get_dummies(model_data['maand'])

model_data = pd.concat([model_data, airline], axis=1)
model_data = pd.concat([model_data, maand], axis=1)

model_data = model_data.drop(['STD', 'FLT', 'STA_STD_ltc', 'ATA_ATD_ltc', 'TAR', 'GAT', 'DL1', 'IX1', 'DL2', 'IX2', 'ACT', 'RWY', 'RWC', 'Identifier', 'verschil', 'dag', 'maatschappij', 'maand', 'LSV', 'Org/Des'], axis=1)

X = model_data.drop('vertraagd', axis=1)
y = model_data['vertraagd']

X.columns = X.columns.astype(str)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

logmodel = lm.LogisticRegression()
logmodel.fit(X_train, y_train)

X_train_pred = logmodel.predict(X_train)
training_data_acc = accuracy_score(y_train, X_train_pred)

st.write('Accuracy score of training data:', training_data_acc)

pred = logmodel.predict(X_test)
test_data_acc = accuracy_score(y_test, pred)
st.write('Accuracy score of test data:', test_data_acc)

# Aantallen krijgen voor de maanden vluchten
monthly_flight_counts = vluchten_copy.groupby(['maand', 'maatschappij']).size().reset_index(name='total_flights')

# Aantallen krijgen voor de vertraagden per maand
monthly_delay_counts = vluchten_copy.groupby(['maand', 'maatschappij'])['vertraagd'].sum().reset_index()


# Samenvoegen en verhouding bepalen
monthly_stats = pd.merge(monthly_flight_counts, monthly_delay_counts, on=['maand', 'maatschappij'])
monthly_stats['delay_rate'] = monthly_stats['vertraagd'] / monthly_stats['total_flights']


# Toekomstige data maken om te voorspellen, per maatschappij en maand
unique_airlines = vluchten_copy['maatschappij'].unique()
future_months = pd.DataFrame()

for maatschappij in unique_airlines:
    airline_future = pd.DataFrame({
        'maand': range(1, 13),  # Months from 1 to 12
        'maatschappij': [maatschappij] * 12  # Repeated airline value for each month
    })
    future_months = pd.concat([future_months, airline_future], ignore_index=True)

# Samenvoegen als een dataframe
future_months = pd.merge(future_months, monthly_flight_counts, on=['maand', 'maatschappij'], how='left')

# Vullen als er geen historische data was met 0
future_months['total_flights'].fillna(0, inplace=True)

# Encoden voor model
future_months_encoded = pd.get_dummies(future_months, columns=['maand', 'maatschappij'])

# Wijzigen van kolomnamen anders kan het model niet toegepast worden
month_columns = [col for col in future_months_encoded.columns if col.startswith('maand_')]
new_month_column_names = {col: col.split('_')[1] for col in month_columns}
future_months_encoded.rename(columns=new_month_column_names, inplace=True)

maatschappij_columns = [col for col in future_months_encoded.columns if col.startswith('maatschappij_')]
new_maatschappij_column_names = {col: col.split('_')[1] for col in maatschappij_columns}
future_months_encoded.rename(columns=new_maatschappij_column_names, inplace=True)

# Voorspel data maken
X_future = future_months_encoded.drop(columns=['total_flights'])

X_future = X_future[X_train.columns]

# Kansen bepalen aan de hand van het model
probabilities = logmodel.predict_proba(X_future)[:, 1] 

# Aantal vertragingen per maand bepalen
future_months['predicted_delays'] = future_months['total_flights'] * probabilities

expected_delays_summary = future_months.groupby('maand')['predicted_delays'].sum().reset_index()

# Maken van plot voor het tonen van het model met de echte waarde ervoor
vluchten_maand = vluchten_copy.copy()

vluchten_maand.set_index('STD', inplace=True)
delayed_counts = vluchten_maand.resample('M')['vertraagd'].sum().reset_index()

year = 2021

# Voeg datum toe aan voorspelling
expected_delays_summary['maand'] = pd.to_datetime(expected_delays_summary['maand'].astype(str) + f'-{year}', format='%m-%Y')

fig6, ax6 = plt.subplots()

sns.lineplot(data=delayed_counts, x='STD', y='vertraagd', marker='o')
sns.lineplot(data=expected_delays_summary, x='maand', y='predicted_delays', marker='o')

plt.legend()
ax6.set_title('Number of Delayed Flights per Month')
ax6.set_xlabel('Date')
ax6.set_ylabel('Number of Delayed Flights')
ax6.tick_params(axis='x', rotation=45)

st.pyplot(fig6)
