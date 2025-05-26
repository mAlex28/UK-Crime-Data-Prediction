import streamlit as st
import joblib
import pandas as pd
import folium
from streamlit_folium import st_folium


regression_model = joblib.load("../notebooks/xgb_model.pkl")
classifier_model = joblib.load('xgb_classifier.pkl')
classifier_le = joblib.load('label_encoder.pkl')


predictions = pd.read_csv("../data/future_predictions.csv")
crime_per_postcode = pd.read_csv("../data/top_crimes_per_postcode.csv")

# Ensure predictions has required columns
required_columns = ['postcode', 'month', 'predicted_crime_count', 'lat', 'lng']
if not all(col in predictions.columns for col in required_columns):
    st.error("The predictions DataFrame must contain 'postcode', 'month', 'predicted_crime_count', 'latitude', and 'longitude' columns.")
    st.stop()

st.set_page_config(page_title="UK Crime Prediction", layout="wide")

st.sidebar.title("Postcode Details")
selected_postcode = st.sidebar.selectbox("Choose a postcode", predictions['postcode'].unique())

# Filter predictions for the selected postcode
selected_data = predictions[predictions['postcode'] == selected_postcode]

# Display details in the sidebar
if not selected_data.empty:
    st.sidebar.write("**Selected Postcode:**", selected_postcode)
    for _, row in selected_data.iterrows():
        st.sidebar.metric(
            label=f"Predicted Crimes for {row['month']}",
            value=int(row['predicted_crime_count'])
        )
else:
    st.sidebar.write("No data available for this postcode.")


# Create Folium map
m = folium.Map(location=[51.454514, -2.587910], zoom_start=14, tiles="OpenStreetMap")  # Centered on Bristol
icon = folium.Icon(prefix="fa", icon="location-pin", color="red")

popup_html = """
<style>
    .popup-container {{
        font-family: Arial, sans-serif;
        background-color: #f9f9f9;
        border: 2px solid #2c3e50;
        border-radius: 8px;
        padding: 15px;
        width: 200px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }}
    .popup-title {{
        font-size: 16px;
        font-weight: bold;
        color: #2c3e50;
        margin-bottom: 10px;
        border-bottom: 1px solid #ddd;
        padding-bottom: 5px;
    }}
    .popup-field {{
        font-size: 14px;
        margin: 5px 0;
        color: #34495e;
    }}
    .popup-field strong {{
        color: #c0392b;
    }}
</style>
<div class='popup-container'>
    <div class='popup-title'>{postcode}</div>
    <div class='popup-field'><strong>Month:</strong> {month}</div>
    <div class='popup-field'><strong>Predicted Crimes:</strong> {crime_count}</div>
</div>
"""

# Add markers for each postcode
for _, row in predictions.iterrows():
    popup_text = popup_html.format(
        postcode=row['postcode'],
        month=row['month'],
        crime_count=int(row['predicted_crime_count'])
    )
    folium.Marker(
        icon=icon,
        location=[row['lat'], row['lng']],
        popup=folium.Popup(popup_text, max_width=250),
        tooltip=row['postcode']
    ).add_to(m)

# Render map in Streamlit
st_folium(m, width=900, height=600)