import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
import joblib

# Load data
regression_model = joblib.load("../notebooks/xgb_model.pkl")
classifier_model = joblib.load('../notebooks/xgb_classifier.pkl')
classifier_le = joblib.load('../notebooks/label_encoder.pkl')
predictions = pd.read_csv("../data/future_predictions.csv")
crime_per_postcode = pd.read_csv("../data/top_crimes_per_postcode.csv")

# Ensure predictions has required columns
required_columns = ['postcode', 'month', 'predicted_crime_count', 'lat', 'lng']
if not all(col in predictions.columns for col in required_columns):
    st.error("The predictions DataFrame must contain 'postcode', 'month', 'predicted_crime_count', 'lat', and 'lng' columns.")
    st.stop()

# Ensure crime_per_postcode has required columns
crime_required_columns = ['postcode', 'anti-social', 'theft', 'violence']
if not all(col in crime_per_postcode.columns for col in crime_required_columns):
    st.error("The crime_per_postcode DataFrame must contain 'postcode', 'anti-social', 'theft', and 'violence' columns.")
    st.stop()

# Determine most likely crime and its probability
crime_types = ['anti-social', 'theft', 'violence']
crime_per_postcode['most_likely_crime'] = crime_per_postcode[crime_types].idxmax(axis=1).str.replace('-', ' ').str.title()
crime_per_postcode['crime_probability'] = crime_per_postcode[crime_types].max(axis=1)

# Merge predictions with crime_per_postcode
merged_data = predictions.merge(
    crime_per_postcode[['postcode', 'most_likely_crime', 'crime_probability']],
    on='postcode',
    how='left'
)

# Handle missing crime data
merged_data['most_likely_crime'] = merged_data['most_likely_crime'].fillna('N/A')
merged_data['crime_probability'] = merged_data['crime_probability'].fillna(0)

# Set page configuration
st.set_page_config(page_title="UK Crime Prediction", layout="wide")

# Create columns for map and details panel
col1, col2 = st.columns([2, 1])  # 2:1 ratio for map and right panel

# Create Folium map
m = folium.Map(location=[51.454514, -2.587910], zoom_start=14, tiles="OpenStreetMap")  # Centered on Bristol
icon = folium.Icon(prefix="fa", icon="location-pin", color="red")

# Custom HTML/CSS for popup
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
        color: #3465e;
    }}
    .popup-field strong {{
        color: #c0392b;
    }}
</style>
<div class='popup-container'>
    <div class='popup-title'>{postcode}</div>
    <div class='popup-field'><strong>Month:</strong> {month}</div>
    <div class='popup-field'><strong>Predicted Crimes:</strong> {crime_count}</div>
    <div class='popup-field'><strong>Most Likely Crime:</strong> {most_likely_crime}</div>
    <div class='popup-field'><strong>Probability:</strong> {crime_probability:.1%}</div>
</div>
"""

# Add markers for each postcode
for _, row in merged_data.iterrows():
    popup_text = popup_html.format(
        postcode=row['postcode'],
        month=row['month'],
        crime_count=int(row['predicted_crime_count']),
        most_likely_crime=row['most_likely_crime'],
        crime_probability=row['crime_probability']
    )
    folium.Marker(
        icon=icon,
        location=[row['lat'], row['lng']],
        popup=folium.Popup(popup_text, max_width=250),
        tooltip=row['postcode']
    ).add_to(m)

# Render map in left column
with col1:
    st.markdown("### Crime Prediction Map")
    output = st_folium(m, width=700, height=500, key="map")

# Right-side panel for crime details
with col2:
    st.markdown("### Crime Details")
    if output and output.get('last_object_clicked') and output['last_object_clicked'].get('tooltip'):
        clicked_postcode = output['last_object_clicked']['tooltip']
        clicked_data = merged_data[merged_data['postcode'] == clicked_postcode]
        if not clicked_data.empty:
            row = clicked_data.iloc[0]  # Take first row if multiple entries
            st.write(f"**Postcode:** {row['postcode']}")
            st.metric(
                label=f"Predicted Crimes for {row['month']}",
                value=int(row['predicted_crime_count'])
            )
            st.write(f"**Crime:** {row['most_likely_crime']}")
            probability_percent = row['crime_probability'] * 100
            st.progress(min(probability_percent / 100, 1.0))  # Progress bar (0 to 1)
            st.write(f"**Probability:** {probability_percent:.1f}%")
            st.markdown("---")
            st.caption("Crimes with less than 10% probability have been ignored for easier classification.")
            # Spacer to align height with map
            st.markdown("<div style='height: 320px;'></div>", unsafe_allow_html=True)
        else:
            st.write("Click a marker on the map to view crime details.")
    else:
        st.write("Click a marker on the map to view crime details.")