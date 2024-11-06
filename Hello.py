import streamlit as st
import folium
import pandas as pd
from streamlit_folium import st_folium

st.set_page_config(
    page_title="Hallo en welkom bij de vliegveldenmap"
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
"""
)


