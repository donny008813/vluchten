import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import sklearn.linear_model as lm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.dates as mdates

# Voeg caching toe voor het laden van de dataset
@st.cache_data
def load_data():
    csv_url = 'https://raw.githubusercontent.com/donny008813/vluchten/main/schedule_airport.csv'
    return pd.read_csv(csv_url)

# Laad de gegevens met caching
vluchten = load_data()

vluchten_copy = vluchten.copy()

# Verwerken van tijdstippen
vluchten_copy['STA_STD_ltc'] = pd.to_datetime(vluchten_copy['STA_STD_ltc'], format='%H:%M:%S').dt.time
vluchten_copy['ATA_ATD_ltc'] = pd.to_datetime(vluchten_copy['ATA_ATD_ltc'], format='%H:%M:%S').dt.time

# Functie om tijd naar seconden om te zetten
def time_to_seconds(time_obj):
    return time_obj.hour * 3600 + time_obj.minute * 60 + time_obj.second

# Function to calculate the difference in total seconds, allowing for negative values
def time_diff_in_seconds(start_time, end_time):
    start_seconds = time_to_seconds(start_time)
    end_seconds = time_to_seconds(end_time)
    
    # Calculate the difference in seconds
    diff_seconds = end_seconds - start_seconds
    
    return diff_seconds

# Verschil in seconden berekenen
vluchten_copy['verschil'] = vluchten_copy.apply(lambda row: time_diff_in_seconds(row['STA_STD_ltc'], row['ATA_ATD_ltc']), axis=1)

# Functie voor het bepalen van vertraging
def vertraagd(x):
    return 1 if x > 900 else 0

vluchten_copy['vertraagd'] = vluchten_copy['verschil'].apply(vertraagd)

# Verwerken van de datum naar extra kenmerken
vluchten_copy['STD'] = pd.to_datetime(vluchten_copy['STD'], format='%d/%m/%Y')
vluchten_copy['dag'] = vluchten_copy['STD'].dt.day
vluchten_copy['maand'] = vluchten_copy['STD'].dt.month
vluchten_copy['dag_van_week'] = vluchten_copy['STD'].dt.dayofweek  # Maandag=0, Zondag=6
vluchten_copy['seizoen'] = vluchten_copy['maand'].apply(lambda x: (x % 12 + 3) // 3)  # 1: winter, 2: lente, 3: zomer, 4: herfst

# Tijdstip van vertrek (uur van de dag)
vluchten_copy['uur_van_vertrek'] = pd.to_datetime(vluchten_copy['STA_STD_ltc'], format='%H:%M:%S').dt.hour

# Functie om maatschappijcode te identificeren
def identify_airline(flight_number):
    return flight_number[:2]

vluchten_copy['maatschappij'] = vluchten_copy['FLT'].apply(identify_airline)

###################################################################################################### EDA Plotten
# Aantal vertraagd
totaal_vertraagd = vluchten_copy['vertraagd'].value_counts()

fig_vertraagd, ax_vertraagd = plt.subplots()
totaal_vertraagd.plot(kind='bar', ax=ax_vertraagd, color=['lightcoral', 'lightgreen'])
ax_vertraagd.set_title('Aantal Vertaagd en niet vertraagd')
ax_vertraagd.set_ylabel('Aantal Vluchten')
ax_vertraagd.set_xlabel('Status')
ax_vertraagd.set_xticklabels(['Niet Vertraagd', 'Vertraagd'], rotation=0)

# Aantal vertraagd per dag van de maand
vertraagd_dag_maand = vluchten_copy.groupby(['dag', 'vertraagd']).size().unstack(fill_value=0)

fig_dag_maand, ax_dag_maand = plt.subplots()
vertraagd_dag_maand.plot(kind='bar', ax=ax_dag_maand, color=['lightcoral', 'lightgreen'])
ax_dag_maand.set_title('Totaal aantal Vluchten per Dag van de Maand')
ax_dag_maand.set_ylabel('Aantal Vluchten')
ax_dag_maand.set_xlabel('Dag van de Maand')
ax_dag_maand.set_xticklabels([str(i) for i in range(1, 32)], rotation=90)  # Labels voor de dagen van de maand
ax_dag_maand.legend(['Niet Vertraagd', 'Vertraagd'])

# Aantal vertraagd per uur van de dag
vertraagd_uur = vluchten_copy.groupby(['uur_van_vertrek', 'vertraagd']).size().unstack(fill_value=0)
fig_uur , ax_uur = plt.subplots()
vertraagd_uur.plot(kind='bar', ax=ax_uur, color=['lightcoral', 'lightgreen'])
ax_uur.set_title('Totaal aantal Vluchten per Uur van de Dag')
ax_uur.set_ylabel('Aantal Vluchten')
ax_uur.set_xlabel('Uur van de Dag')
ax_uur.set_xticks(range(0, 24))
ax_uur.set_xticklabels([str(i) for i in range(0, 24)], rotation=0)  # Labels voor de uren van de dag
ax_uur.legend(['Niet Vertraagd', 'Vertraagd'])

# Totaal aantal vertraagde en niet-vertraagde vluchten per maand van het jaar
totaal_vertraagd_per_maand = vluchten_copy.groupby(['maand', 'vertraagd']).size().unstack(fill_value=0)

fig_maand, ax_maand = plt.subplots()
totaal_vertraagd_per_maand.plot(kind='bar', ax=ax_maand, color=['lightcoral', 'lightgreen'])
ax_maand.set_title('Totaal aantal Vluchten per Maand van het Jaar')
ax_maand.set_ylabel('Aantal Vluchten')
ax_maand.set_xlabel('Maand')
ax_maand.set_xticks(range(0, 12))
ax_maand.set_xticklabels(['Jan', 'Feb', 'Mrt', 'Apr', 'Mei', 'Jun', 'Jul', 'Aug', 'Sep', 'Okt', 'Nov', 'Dec'], rotation=45)
ax_maand.legend(['Niet Vertraagd', 'Vertraagd'])

# Aantal vertraagde en niet-vertraagde vluchten per seizoen tellen
vertraagd_aantal = vluchten_copy.groupby(['seizoen', 'vertraagd']).size().unstack(fill_value=0)

# Plotten
fig, ax = plt.subplots()
vertraagd_aantal.plot(kind='bar', ax=ax, color=['lightcoral', 'lightgreen'])
ax.set_title('Aantal Vertraagde en Niet-Vertraagde Vluchten per Seizoen')
ax.set_ylabel('Aantal Vluchten')
ax.set_xlabel('Seizoen')
ax.set_xticklabels(['Lente', 'Zomer', 'Herfst', 'Winter'], rotation=0)  # Labels voor de seizoenen
ax.legend(['Niet Vertraagd', 'Vertraagd'])

# Aantal vertraagde vluchten per dag van de week
vertraagd_aantal_dag = vluchten_copy.groupby(['dag_van_week', 'vertraagd']).size().unstack(fill_value=0)

# Plotten
fig_dagen, ax_dagen = plt.subplots()
vertraagd_aantal_dag.plot(kind='bar', ax=ax_dagen, color=['lightcoral', 'lightgreen'])
ax_dagen.set_title('Aantal vertraagde en niet vertraagde vluchten per dag van de week')
ax_dagen.set_ylabel('Aantal Vluchten')
ax_dagen.set_xlabel('Dag van de week')
ax_dagen.set_xticklabels(['Ma', 'Di', 'Wo', 'Do', 'Vrij', 'Za', 'Zo'], rotation=0)
ax_dagen.legend(['Niet Vertraagd', 'Vertraagd'])

# Hist vertraagd maatschappij
flight_counts = vluchten_copy['maatschappij'].value_counts().nlargest(10).index

# Filtering DataFrame for top 10 airlines
top_10_df = vluchten_copy[vluchten_copy['maatschappij'].isin(flight_counts)]

fig4, ax4 = plt.subplots()
sns.countplot(data=top_10_df, x='maatschappij', hue='vertraagd')
ax4.set_title('Count of Delays and Non-Delays per Airline')
ax4.set_ylabel('Count')
ax4.set_xlabel('Airline')

# Group by date and count the number of delayed flights
delayed_counts = vluchten_copy.groupby('STD')['vertraagd'].sum().reset_index()

fig5, ax5 = plt.subplots()

sns.barplot(x='STD', y='vertraagd', data=delayed_counts)

ax5.set_title('Number of Delayed Flights per Day')
ax5.set_xlabel('Date')
ax5.set_ylabel('Number of Delayed Flights')
ax5.tick_params(axis='x', rotation=45)
ax5.xaxis.set_major_locator(mdates.YearLocator())  # Set major ticks for each year
plt.tight_layout()

# Weergave in Streamlit
st.title('Vluchten Analyse')
st.write(vluchten_copy)
st.pyplot(fig_vertraagd)
plt.close(fig_vertraagd)
st.pyplot(fig) # Seizoenen
plt.close(fig)
st.pyplot(fig_maand)
plt.close(fig_maand)
st.pyplot(fig_dag_maand)
plt.close(fig_dag_maand)
st.pyplot(fig_dagen)
plt.close(fig_dagen)
st.pyplot(fig_uur)
plt.close(fig_uur)
st.pyplot(fig4) # Top 10 Maatschappij
plt.close(fig4)

# Dropdown menu voor selectie van vliegtuigmaatschappij
maatschappij_selectie = st.selectbox('Selecteer een vliegtuigmaatschappij', df['maatschappij'].unique())

# Filter DataFrame op geselecteerde vliegtuigmaatschappij
df_geselecteerd = vluchten_copy[vluchten_copy['maatschappij'] == maatschappij_selectie]

# Totaal aantal vertraagde en niet-vertraagde vluchten voor de geselecteerde maatschappij per maand
totaal_vertraagd_per_maand = df_geselecteerd.groupby(['maand', 'vertraagd']).size().unstack(fill_value=0)

# Plotten
st.subheader(f'Totaal aantal Vluchten voor {maatschappij_selectie} per Maand van het Jaar: Vertraagd vs Niet-Vertraagd')

fig_maat, ax_maat = plt.subplots()
totaal_vertraagd_per_maand.plot(kind='bar', ax=ax_maat, color=['lightcoral', 'lightgreen'])
ax_maat.set_title(f'Totaal aantal Vluchten voor {maatschappij_selectie} per Maand van het Jaar')
ax_maat.set_ylabel('Aantal Vluchten')
ax_maat.set_xlabel('Maand')
ax_maat.set_xticks(range(0, 12))
ax_maat.set_xticklabels(['Jan', 'Feb', 'Mrt', 'Apr', 'Mei', 'Jun', 'Jul', 'Aug', 'Sep', 'Okt', 'Nov', 'Dec'], rotation=45)
ax_maat.legend(['Niet Vertraagd', 'Vertraagd'])
st.pyplot(fig_maat)
plt.close(fig_maat)

st.pyplot(fig5) # Hele dataframe
plt.close(fig5)

########################################################################################################### Model
# One-hot encoding voor categorische variabelen
airline = pd.get_dummies(vluchten_copy['maatschappij'])
maand = pd.get_dummies(vluchten_copy['maand'], prefix='maand')
dag_van_week = pd.get_dummies(vluchten_copy['dag_van_week'], prefix='dag_van_week')
seizoen = pd.get_dummies(vluchten_copy['seizoen'], prefix='seizoen')
uur_van_vertrek = pd.get_dummies(vluchten_copy['uur_van_vertrek'], prefix='uur')

# Toevoegen van nieuwe features aan de data
model_data = pd.concat([vluchten_copy, airline, maand, dag_van_week, seizoen], axis=1)
model_data = pd.concat([model_data, uur_van_vertrek], axis=1)

# Onnodige kolommen verwijderen
model_data = model_data.drop(['STD', 'FLT', 'STA_STD_ltc', 'ATA_ATD_ltc', 'TAR', 'GAT', 'DL1', 'IX1', 'DL2', 'IX2',
                              'ACT', 'RWY', 'RWC', 'Identifier', 'verschil', 'dag', 'maatschappij', 'maand', 'LSV',
                              'Org/Des'], axis=1)

# Modeldata voorbereiden
X = model_data.drop('vertraagd', axis=1)
y = model_data['vertraagd']

X.columns = X.columns.astype(str)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# Model trainen
logmodel = lm.LogisticRegression(max_iter=500)
logmodel.fit(X_train, y_train)

# Resultaten printen
X_train_pred = logmodel.predict(X_train)
training_data_acc = accuracy_score(y_train, X_train_pred)
st.write('Accuracy score of training data:', training_data_acc)

pred = logmodel.predict(X_test)
test_data_acc = accuracy_score(y_test, pred)
st.write('Accuracy score of test data:', test_data_acc)

# Toekomstige data maken om te voorspellen per maatschappij, maand, uur van de dag, dag van de week en seizoen
future_data = pd.DataFrame()

# Genereer combinaties van maand, uur, dag van de week en seizoen voor alle maatschappijen
unique_airlines = vluchten_copy['maatschappij'].unique()
for maatschappij in unique_airlines:
    for maand in range(1, 13):  # Maanden 1 tot 12
        for uur in range(24):  # Uren 0 tot 23
            for dag_van_week in range(7):  # Dagen van de week 0 tot 6
                for seizoen in range(1, 5):  # Seizoenen 1 tot 4
                    row = {
                        'maand': maand,
                        'uur_van_vertrek': uur,
                        'dag_van_week': dag_van_week,
                        'seizoen': seizoen,
                        'maatschappij': maatschappij
                    }
                    future_data = pd.concat([future_data, pd.DataFrame([row])], ignore_index=True)

# One-hot encoding van de toekomstige data
future_data_encoded = pd.get_dummies(future_data, columns=['maand', 'dag_van_week', 'seizoen', 'maatschappij', 'uur_van_vertrek'])

# Zorg dat de kolommen overeenkomen met het getrainde model
future_data_encoded = future_data_encoded.reindex(columns=X_train.columns, fill_value=0)

# Voorspellen van het aantal vertraagde vluchten
future_data_encoded['predicted_delays'] = logmodel.predict_proba(future_data_encoded)[:, 1]

# Samenvatting van voorspellingen per maand maken
future_data_encoded['maand'] = future_data_encoded['maand'].apply(lambda x: x if isinstance(x, int) else 0)
predicted_delays_summary = future_data_encoded.groupby('maand')['predicted_delays'].sum().reset_index()

# Plot maken van de voorspellingen
fig_voorspel, ax_voorspel = plt.subplots()
sns.lineplot(data=predicted_delays_summary, x='maand', y='predicted_delays', marker='o')
ax_voorspel.set_title('Verwacht aantal vertraagde vluchten per maand')
ax_voorspel.set_xlabel('Maand')
ax_voorspel.set_ylabel('Verwacht aantal vertraagde vluchten')
plt.xticks(range(1, 13), ['Jan', 'Feb', 'Mrt', 'Apr', 'Mei', 'Jun', 'Jul', 'Aug', 'Sep', 'Okt', 'Nov', 'Dec'])
plt.grid(True)

st.pyplot(fig_voorspel)

