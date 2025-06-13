import streamlit as st
import joblib
import pandas as pd
import folium
from streamlit_folium import st_folium


regression_model = joblib.load("../notebooks/xgb_model.pkl")
classifier_model = joblib.load('../notebooks/xgb_classifier.pkl')
classifier_le = joblib.load('../notebooks/label_encoder.pkl')

predictions = pd.read_csv("../data/future_predictions.csv")
crime_per_postcode = pd.read_csv("../data/top_crimes_per_postcode.csv")

# Ensure predictions has required columns
required_columns = ['postcode', 'month', 'predicted_crime_count', 'lat', 'lng']
if not all(col in predictions.columns for col in required_columns):
    st.error("The predictions DataFrame must contain 'postcode', 'month', 'predicted_crime_count', 'latitude', and 'longitude' columns.")
    st.stop()

st.set_page_config(page_title="UK Crime Prediction", layout="wide")

st.title("Postcode Details")
selected_postcode = st.selectbox("Choose a postcode", predictions['postcode'].unique())

# Filter predictions for the selected postcode
selected_data = predictions[predictions['postcode'] == selected_postcode]

# Display details in the sidebar
if not selected_data.empty:
    st.write("**Selected Postcode:**", selected_postcode)
    for _, row in selected_data.iterrows():
        st.metric(
            label=f"Predicted Crimes for {row['month']}",
            value=int(row['predicted_crime_count'])
        )
else:
    st.sidebar.write("No data available for this postcode.")

if 'clicked_postcode' not in st.session_state:
    st.session_state.clicked_postcode = selected_postcode

col1, col2 = st.columns([3, 1])

with col1:
    # Create Folium map
    m = folium.Map(location=[51.454514, -2.587910], zoom_start=14, tiles="OpenStreetMap")
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

    # Add markers
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

    map_data = st_folium(m, width=900, height=600, key="map", returned_objects=["last_clicked"])

    if map_data['last_clicked'] and 'lat' in map_data['last_clicked']:
        clicked_postcode = next((row['postcode'] for _, row in predictions.iterrows() if abs(row['lat'] - map_data['last_clicked']['lat']) < 0.001 and abs(row['lng'] - map_data['last_clicked']['lng']) < 0.001), st.session_state.clicked_postcode)
        st.session_state.clicked_postcode = clicked_postcode
    else:
        clicked_postcode = st.session_state.clicked_postcode



with col2:
    # Filter crime percentages for the clicked or selected postcode
    crime_data = crime_per_postcode[crime_per_postcode['postcode'] == st.session_state.clicked_postcode]
    if not crime_data.empty:
        st.write("**Crime Percentages:**")
        crime_row = crime_data.iloc[0]  # Take the first row if multiple entries
        st.metric(label="Anti-social", value=f"{int(crime_row['anti-social'] * 100)}%")
        st.metric(label="Theft", value=f"{int(crime_row['theft'] * 100)}%")
        st.metric(label="Violence", value=f"{int(crime_row['violence'] * 100)}%")
        st.caption("Crimes with less than 10% probability have been ignored for easier classification.")
    else:
        st.write("No crime percentage data available.")