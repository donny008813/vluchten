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

# Data inladen
csv_url = 'https://raw.githubusercontent.com/donny008813/vluchten/main/schedule_airport.csv'
vluchten = pd.read_csv(csv_url)

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
    return 1 if x > 0 else 0

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

# One-hot encoding voor categorische variabelen
airline = pd.get_dummies(vluchten_copy['maatschappij'])
maand = pd.get_dummies(vluchten_copy['maand'], prefix='maand')
dag_van_week = pd.get_dummies(vluchten_copy['dag_van_week'], prefix='dag_van_week')
seizoen = pd.get_dummies(vluchten_copy['seizoen'], prefix='seizoen')

# Toevoegen van nieuwe features aan de data
model_data = pd.concat([vluchten_copy, airline, maand, dag_van_week, seizoen], axis=1)

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

# Voor toekomstige voorspellingen met de nieuwe variabelen
future_months = pd.DataFrame()

# Unieke combinaties van uur van vertrek, dag van de week en seizoen
unique_airlines = vluchten_copy['maatschappij'].unique()
for maatschappij in unique_airlines:
    for uur in range(24):  # Alle uren van de dag
        for dag_van_week in range(7):  # Alle dagen van de week
            for seizoen in range(1, 5):  # Alle seizoenen
                future_data = pd.DataFrame({
                    'uur_van_vertrek': [uur],
                    'dag_van_week': [dag_van_week],
                    'seizoen': [seizoen],
                    'maatschappij': [maatschappij]
                })
                future_months = pd.concat([future_months, future_data], ignore_index=True)

# Encoderen van de nieuwe data
future_months = pd.get_dummies(future_months, columns=['dag_van_week', 'seizoen', 'maatschappij'])

# Kolomnamen aanpassen voor consistentie met het model
future_months = future_months.reindex(columns=X_train.columns, fill_value=0)

# Voorspellen van vertragingen
probabilities = logmodel.predict_proba(future_months)[:, 1]
future_months['predicted_delays'] = probabilities

# Samenvatting per uur/dag/seizoen maken voor visualisatie
predicted_delays_summary = future_months.groupby(['uur_van_vertrek'])['predicted_delays'].sum().reset_index()

# Plotten van de voorspelde vertragingen
fig, ax = plt.subplots()

sns.lineplot(data=predicted_delays_summary, x='uur_van_vertrek', y='predicted_delays', marker='o')
ax.set_title('Verwachte vertragingen per uur van de dag')
ax.set_xlabel('Uur van vertrek')
ax.set_ylabel('Aantal verwachte vertragingen')

plt.xticks(range(24))  # Labels voor elk uur van de dag
plt.grid(True)
st.pyplot(fig)

# Gemiddelde vertragingen per uur van vertrek plotten
fig1, ax1 = plt.subplots()
sns.barplot(data=vluchten_copy, x='uur_van_vertrek', y='vertraagd', estimator=np.mean, ci=None)
ax1.set_title('Gemiddelde kans op vertraging per uur van vertrek')
ax1.set_xlabel('Uur van vertrek')
ax1.set_ylabel('Gemiddelde kans op vertraging')
plt.xticks(range(24))  # Labels voor elk uur van de dag
plt.grid(True)
st.pyplot(fig1)

# Gemiddelde vertragingen per dag van de week plotten
fig2, ax2 = plt.subplots()
sns.barplot(data=vluchten_copy, x='dag_van_week', y='vertraagd', estimator=np.mean, ci=None)
ax2.set_title('Gemiddelde kans op vertraging per dag van de week')
ax2.set_xlabel('Dag van de week (0=Ma, 6=Zo)')
ax2.set_ylabel('Gemiddelde kans op vertraging')
plt.grid(True)
st.pyplot(fig2)

# Gemiddelde vertragingen per seizoen plotten
fig3, ax3 = plt.subplots()
sns.barplot(data=vluchten_copy, x='seizoen', y='vertraagd', estimator=np.mean, ci=None)
ax3.set_title('Gemiddelde kans op vertraging per seizoen')
ax3.set_xlabel('Seizoen (1=Winter, 2=Lente, 3=Zomer, 4=Herfst)')
ax3.set_ylabel('Gemiddelde kans op vertraging')
plt.grid(True)
st.pyplot(fig3)
