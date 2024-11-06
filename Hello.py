import streamlit as st
import folium
import pandas as pd
from streamlit_folium import st_folium

st.set_page_config(
    page_title="Hallo en welkom bij de verbetering van case 3"
)

st.write("# Welkom bij het dashboard van de verbeteringen van case 3.")

st.markdown("""
In dit dashboard zal voor Visual Analytics de verbeteropdracht worden getoond. Het was het de opdracht om een case te verbeteren en
wij hebben gekozen om case 3 te verbeteren. 

Deze case hebben wij gekozen omdat het onderwerp ons aansprak en er nog ruimte was voor verbetering van de EDA en het opstellen van een model. 

Eerst zal er gekeken worden naar de kaart die voor deze opdracht is opgesteld en hoe wij deze hebben verbeterd. Daarna zal er gekeken worden naar het onderzoek wat is gedaan
naar het aantal vluchten en hoe wij deze visualisaties hebben verbeterd. Ook nog een kleine verbetering in de visualisatie voor het vergelijken van het opstijgen en landen van 
verschillende vluchten. En ten slotte wordt het oude model getoond en hoe wij deze mogelijk hebben verbeterd door extra features te maken en te kijken of deze invloed hebben. Met 
de nieuwe scores van het model worden samen getoond met die van het oude model.

Bronnen die zijn gebruikt:

- Data beschikbaar gesteld door de HvA: https://dlo.mijnhva.nl/d2l/le/content/614352/viewContent/2210753/View

- Data beschikbaar gesteld door de HvA: https://dlo.mijnhva.nl/d2l/le/content/614352/viewContent/2210755/View

- Lessen van Datacamp

- Hoor en werkcolleges van Minor Data Science Introduction to Data Science

- ChatGPT
"""
)


