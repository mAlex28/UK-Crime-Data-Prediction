import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import folium
from streamlit_folium import st_folium
import joblib
import path
import sys

dir = path.Path(__file__).abspath()
sys.path.append.path(dir.parent.parent)

# Load models
regression_model = joblib.load("assets/models/xgb_model.pkl")
classifier_model = joblib.load("assets/models/xgb_classifier.pkl")
classifier_le = joblib.load("assets/models/label_encoder.pkl")

# Load predictions
future_predictions = pd.read_csv("assets/predictions/future_predictions.csv")
top_crime_per_postcode = pd.read_csv("assets/predictions/top_crimes_per_postcode.csv")
risk_level_per_postcode = pd.read_csv("assets/predictions/risk_level_per_postcode.csv")

# Ensure predictions have required columns
required_columns = {
    "top_crime_per_postcode": ['postcode', 'lat', 'lng', 'crime_count_per_month', 'anti-social', 'theft', 'violence'],
    "future_predictions": ['postcode', 'month', 'predicted_crime_count', 'lat', 'lng'],
    "risk_level_per_postcode": ['postcode', 'month', 'predicted_crime_count', 'lat', 'lng', 'risk', 'colour']
}

for name, cols in required_columns.items():
    df_check = eval(name)
    if not all(col in df_check.columns for col in cols):
        st.error(f"{name}.csv must contain: {', '.join(cols)}")
        st.stop()

# Merge datasets into one
@st.cache_data
def load_crime_data(selected_month=None):

    df_future = future_predictions.copy()
    df_future = future_predictions.copy()
    if selected_month:
        df_future = df_future[df_future['month'] == selected_month]

    df = df_future.merge(
        top_crime_per_postcode[['postcode', 'anti-social', 'theft', 'violence', 'crime_count_per_month']],
        on='postcode', how='left'
    ).merge(
        risk_level_per_postcode[['postcode', 'risk', 'colour']],on='postcode', how='left'
    )

    df.rename(columns={'lng': 'lon'}, inplace=True)
    df['city'] = 'Bristol'
    df['total_crimes'] = df['predicted_crime_count']
    df['sexual'] = 0
    df['other'] = 0

    # Most likely crime
    crime_types = ['anti-social', 'theft', 'violence']
    df['most_likely_crime'] = df[crime_types].idxmax(axis=1).str.replace('-', ' ').str.title()
    df['crime_probability'] = df[crime_types].max(axis=1)

    df = df.drop_duplicates(subset='postcode')

    return df

# Risk Colours
@st.cache_data
def get_risk_colour(risk):
    colour_map = {
        'High': "#dd5858ba",
        'Medium': "#cf9022bc",
        'Low': "#2ae15bbf"
    }
    return colour_map.get(risk, '#6b7280')

# Crime map
def create_crime_map(df, selected_risk, selected_postcode=None):
    if selected_risk != 'All':
        df = df[df['risk'] == selected_risk]

    # Default location: centre of Bristol
    default_location = [51.4545, -2.5879]
    default_zoom = 13

    if selected_postcode:
        focus_match = df[df['postcode'].str.startswith(selected_postcode, na=False)]

        if not focus_match.empty:
            focus_row = focus_match.iloc[0]
            map_location = [focus_row['lat'], focus_row['lon']]
            map_zoom = 16
        else:
            map_location = default_location
            map_zoom = default_zoom
    else:
        map_location = default_location
        map_zoom = default_zoom

    # Create map with dynamic location
    m = folium.Map(location=map_location, zoom_start=map_zoom)

    # Add markers
    for _, row in df.iterrows():
        colour = get_risk_colour(row['risk'])
        popup = f"""
            <b>{row['postcode']}</b><br>
            Crimes: {int(row['predicted_crime_count'])}<br>
            Risk: {row['risk']}<br>
            Most Likely: {row['most_likely_crime']} ({row['crime_probability']:.0%})
        """
        if selected_postcode and row['postcode'] == selected_postcode:
            folium.Marker(
                [row['lat'], row['lon']],
                popup=popup,
                tooltip=row['postcode'],
                icon=folium.Icon(color='red', icon='star')
            ).add_to(m)
        else:
            folium.CircleMarker(
                location=[row['lat'], row['lon']],
                radius=10,
                color=colour,
                fill=True,
                fill_color=colour,
                fill_opacity=0.15,
                opacity=1,
                stroke=False,
                popup=popup,
                tooltip=row['postcode']
            ).add_to(m)
    return m


# Page config
st.set_page_config(page_title="Bristol Crime Prediction Dashboard", layout="wide")

# Import font awesome icons
st.markdown(
    """
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    """,
    unsafe_allow_html=True
)

st.markdown("""
    <style>
    .main-header {
        background: #f8f9fa;
        color: #212529;
        padding: 0.4rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
    }
            
    .main-header h1 {
        color: #212529;
        margin: 0;
    }

    .main-header p, .main-header a {
        color: #495057; 
        margin: 0;
        margin-top: 0.5rem;
    }
            
    a:visited {
        text-decoration: none;
    }
            
    a:hover{
        color: #000000;
    }
    
    .risk-high { border-left: 4px solid #ef4444; }
    .risk-medium { border-left: 4px solid #f59e0b; }
    .risk-low { border-left: 4px solid #10b981; }
    .crime-metric {
        background: #f8fafc;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)


# Main App
def main():
    st.info('Currently, crime predictions are available for Bristol. More cities are on the way!')

    # Header
    st.markdown("""
    <div class="main-header">
        <h1>Crime Prediction Dashboard</h1>
        <p>Using AI to predict crimes in UK cities</p>
        <a href="https://github.com/mAlex28/UK-Crime-Data-Prediction" style="text-decoration:none;"><i class="fa-brands fa-github" style="color: #495057;"></i> View Source </a>
    </div>
    """, unsafe_allow_html=True)

    # Get first and last month of data available
    all_dates = pd.to_datetime(load_crime_data()['month'].dropna().unique())
    all_dates = sorted(all_dates)
    first_month = all_dates[0].strftime('%B %Y')
    last_month = all_dates[-1].strftime('%B %Y')

    with st.sidebar:
        # Month selection
        month_options = sorted(future_predictions['month'].dropna().unique())
        selected_month = st.selectbox("Select Month", sorted(month_options), index=0)

        # Load data
        df = load_crime_data(selected_month)

        # Search Postcode
        search = st.text_input("Search Postcode..", placeholder="e.g., BS1 1BU")
        selected_postcode = None
        if search:
            matches = df[df['postcode'].str.contains(search, case=False)]
            if not matches.empty:
                selected_postcode = st.selectbox("Select Postcode", matches['postcode'].tolist())
            else:
                st.warning("No matches found.")

        # Risk filter
        selected_risk = st.selectbox("Risk level", ["Low", "Medium", "High"])

        st.divider()

        st.header("Overall Stats")
        st.metric("Covered Postcodes", df['postcode'].nunique())
        st.metric("Avg Predicted Crimes", f"{df['predicted_crime_count'].mean():.0f}")
        st.metric("High Risk Areas", df[df['risk'] == 'High'].shape[0])


    # Main visual section
    col1, col2 = st.columns(spec=[2,1], vertical_alignment="top")

    # with col1:
        # st.subheader("Bristol Overall")
        # totals = {
        #     'Theft': (df['theft'] * df['total_crimes']).sum(),
        #     'Violence': (df['violence'] * df['total_crimes']).sum(),
        #     'Anti-social': (df['anti-social'] * df['total_crimes']).sum()
        # }
        # for crime, val in sorted(totals.items(), key=lambda x: x[1], reverse=True):
        #     st.metric(crime, f"{int(val)}")

    with col1:
        crime_map = create_crime_map(df, selected_risk, selected_postcode)
        st_folium(crime_map, height=500, width=700)

    with col2:
        if selected_postcode:
            pd_data = df[df['postcode'] == selected_postcode].iloc[0]
            st.markdown(f"""
                <div class="crime-metric risk-{pd_data['risk'].lower()}">
                    <h3>{pd_data['postcode']} - Bristol</h3>
                    <h2>{int(pd_data['total_crimes'])} predicted crimes</h2>
                    <p><strong>Risk:</strong> {pd_data['risk']}</p>
                </div>
            """, unsafe_allow_html=True)

            st.subheader("Crime Breakdown")
            for crime in ['theft', 'violence', 'anti-social']:
                st.metric(
                    label=crime.title(),
                    value=f"{pd_data[crime]:.0%} chance",
                    delta=f"{(pd_data[crime] * pd_data['total_crimes']):.0f} est. cases"
                )

            # Pie chart
            fig = px.pie(
                names=['Theft', 'Violence', 'Anti-social'],
                values=[pd_data['theft'], pd_data['violence'], pd_data['anti-social']],
                title="Most Likely Crime Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)

        else:
            st.info("Select a postcode to view detailed statistics")

      
    

    st.markdown(f"""
    <hr>
    <div style='text-align:center; color: #6b7280'>
        ⚠️ This is a learning project — predictions are generated using AI models and historical data.<br>
    They may not be accurate. <br>
        <strong>Prediction period: {first_month} to {last_month}</strong>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()