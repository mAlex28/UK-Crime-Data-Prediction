import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import folium
from streamlit_folium import st_folium
import joblib

# Load models
regression_model = joblib.load("../assets/models/xgb_model.pkl")
classifier_model = joblib.load('../assets/models/xgb_classifier.pkl')
classifier_le = joblib.load('../assets/models/label_encoder.pkl')

# Load predictions
predictions = pd.read_csv("../assets/predictions/future_predictions.csv")
crime_per_postcode = pd.read_csv("../assets/predictions/top_crimes_per_postcode.csv")
risk_level_per_postcode = pd.read_csv("../assets/predictions/risk_level_per_postcode.csv")

# Ensure predictions have required columns
required_columns = {
    'predictions': ['postcode', 'month', 'predicted_crime_count', 'lat', 'lng'],
    'crime_per_postcode': ['postcode', 'anti-social', 'theft', 'violence'],
    'risk_level_per_postcode': ['postcode', 'predicted_crime_count', 'risk']
}
for name, cols in required_columns.items():
    df_check = eval(name)
    if not all(col in df_check.columns for col in cols):
        st.error(f"{name}.csv must contain: {', '.join(cols)}")
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

# Page configuration
st.set_page_config(
    page_title="UK Crime Prediction Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS
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
            
    .crime-metric {
        background: #f8fafc;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #3b82f6;
        margin: 0.5rem 0;
    }
    .risk-high { border-left-color: #ef4444 !important; }
    .risk-medium { border-left-color: #f59e0b !important; }
    .risk-low { border-left-color: #10b981 !important; }
</style>
""", unsafe_allow_html=True)

st.markdown(
    """
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    """,
    unsafe_allow_html=True
)

# Mock data for demonstration
@st.cache_data
def load_crime_data():
    """Load mock crime prediction data"""
    data = {
        'postcode': ['SW1A 1AA', 'M1 1AA', 'B1 1AA', 'LS1 1AA', 'E1 6AN', 'G1 1AA', 'CF10 1AA', 'BT1 1AA'],
        'city': ['London', 'Manchester', 'Birmingham', 'Leeds', 'London', 'Glasgow', 'Cardiff', 'Belfast'],
        'lat': [51.5014, 53.4808, 52.4862, 53.8008, 51.5155, 55.8642, 51.4816, 54.5973],
        'lon': [-0.1419, -2.2426, -1.8904, -1.5491, -0.0922, -4.2518, -3.1791, -5.9301],
        'total_crimes': [45, 32, 28, 15, 38, 25, 22, 18],
        'theft': [18, 12, 10, 6, 15, 9, 8, 7],
        'violence': [15, 11, 9, 4, 12, 8, 7, 6],
        'sexual': [5, 3, 4, 2, 4, 3, 3, 2],
        'other': [7, 6, 5, 3, 7, 5, 4, 3],
        'population': [1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000],
        'risk_level': ['high', 'medium', 'medium', 'low', 'high', 'medium', 'low', 'low']
    }
    return pd.DataFrame(data)

@st.cache_data
def get_risk_color(risk_level):
    """Get color based on risk level"""
    colors = {'low': '#10b981', 'medium': '#f59e0b', 'high': '#ef4444'}
    return colors.get(risk_level, '#6b7280')

def create_crime_map(df, selected_risk_level, selected_postcode=None):
    """Create an interactive map with crime data"""
    # Filter data based on risk level
    if selected_risk_level != 'all':
        df_filtered = df[df['risk_level'] == selected_risk_level]
    else:
        df_filtered = df
    
    # Create base map centered on UK
    m = folium.Map(location=[54.5, -3], zoom_start=14,  tiles="OpenStreetMap")
    icon = icon = folium.Icon(prefix="fa", icon="location-pin", color="red")
    
    # Add markers for each postcode
    for idx, row in df_filtered.iterrows():
        color = get_risk_color(row['risk_level'])
        
        # Create popup content
        popup_content = f"""
        <b>{row['postcode']} - {row['city']}</b><br>
        Total Crimes: {row['total_crimes']}/1000<br>
        Risk Level: {row['risk_level'].title()}<br>
        Theft: {row['theft']}<br>
        Violence: {row['violence']}<br>
        Sexual: {row['sexual']}<br>
        Other: {row['other']}
        """
        
        # Highlight selected postcode
        if selected_postcode and row['postcode'] == selected_postcode:
            folium.Marker(
                [row['lat'], row['lon']],
                popup=popup_content,
                tooltip=f"{row['postcode']} - {row['city']}",
                icon=folium.Icon(color='red', icon='star')
            ).add_to(m)
        else:
            folium.CircleMarker(
                [row['lat'], row['lon']],
                radius=row['total_crimes'] / 2,
                popup=popup_content,
                tooltip=f"{row['postcode']} - {row['city']}",
                color=color,
                fillColor=color,
                fillOpacity=0.7
            ).add_to(m)
    
    return m

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
    
    # Load data
    df = load_crime_data()
    

    # Sidebar for controls and statistics
    with st.sidebar:

        # Search functionality
        st.header("🔍 Search")
        search_query = st.text_input(
            "Search Postcode or City",
            placeholder="e.g., SW1A 1AA or London"
        )
        
        # Find matching postcodes/cities
        if search_query:
            matches = df[
                df['postcode'].str.contains(search_query, case=False) |
                df['city'].str.contains(search_query, case=False)
            ]
            
            if not matches.empty:
                selected_postcode = st.selectbox(
                    "Select from matches:",
                    options=matches['postcode'].tolist(),
                    format_func=lambda x: f"{x} - {df[df['postcode']==x]['city'].iloc[0]}"
                )
            else:
                selected_postcode = None
                st.warning("No matches found")
        else:
            selected_postcode = None
        
        st.divider()


        # Month selection
        months = [
            ("2025-01", "January 2025"),
            ("2025-02", "February 2025")
        ]
        selected_month = st.selectbox(
            "📅 Select Prediction Month",
            options=[m[0] for m in months],
            format_func=lambda x: dict(months)[x],
            index=0
        )
        
        # Risk level filter
        risk_levels = ['all', 'low', 'medium', 'high']
        selected_risk = st.selectbox(
            "🎯 Filter by Risk Level",
            options=risk_levels,
            format_func=lambda x: x.title() if x != 'all' else 'All Levels',
            index=0
        )
        
        st.divider()
        
        
        # Crime Statistics Panel
        st.header("📊 Crime Statistics")
        
        if selected_postcode:
            postcode_data = df[df['postcode'] == selected_postcode].iloc[0]
            
            # Display main metrics
            st.markdown(f"""
            <div class="crime-metric risk-{postcode_data['risk_level']}">
                <h3>{postcode_data['postcode']} - {postcode_data['city']}</h3>
                <h2>{postcode_data['total_crimes']} crimes per 1,000 population</h2>
                <p><strong>Risk Level:</strong> {postcode_data['risk_level'].title()}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Crime breakdown
            st.subheader("Crime Breakdown")
            
            crime_types = ['theft', 'violence', 'sexual', 'other']
            crime_data = [(crime, postcode_data[crime]) for crime in crime_types]
            crime_data.sort(key=lambda x: x[1], reverse=True)
            
            for crime_type, count in crime_data:
                percentage = (count / postcode_data['total_crimes']) * 100
                st.metric(
                    label=crime_type.title(),
                    value=f"{count} cases",
                    delta=f"{percentage:.1f}%"
                )
            
            # Crime distribution chart
            fig_pie = px.pie(
                values=[count for _, count in crime_data],
                names=[crime_type.title() for crime_type, _ in crime_data],
                title="Crime Distribution",
                color_discrete_sequence=['#ef4444', '#f59e0b', '#8b5cf6', '#06b6d4']
            )
            fig_pie.update_layout(height=300)
            st.plotly_chart(fig_pie, use_container_width=True)
            
        else:
            st.info("Select a postcode on the map or search to view detailed statistics")
    
    # Main content area
    col1, col2, col3 = st.columns(spec=[1,3,1], vertical_alignment="top")

    with col1:
        st.header("Bristol")
        
        # Overall statistics
        total_locations = len(df)
        avg_crimes = df['total_crimes'].mean()
        high_risk_count = len(df[df['risk_level'] == 'high'])
        
        st.metric("Population", total_locations)
        st.metric("Average Crime Rate", f"{avg_crimes:.1f}/1000")
        st.metric("High Risk Areas", high_risk_count)
        
        # # Risk level distribution
        # risk_counts = df['risk_level'].value_counts()
        # fig_bar = px.bar(
        #     x=risk_counts.index,
        #     y=risk_counts.values,
        #     title="Risk Level Distribution",
        #     color=risk_counts.index,
        #     color_discrete_map={
        #         'low': '#10b981',
        #         'medium': '#f59e0b',
        #         'high': '#ef4444'
        #     }
        # )
        # fig_bar.update_layout(height=300, showlegend=False)
        # st.plotly_chart(fig_bar, use_container_width=True)
        
        
    
    with col2:
        # Display selected postcode info if available
        if selected_postcode:
            postcode_data = df[df['postcode'] == selected_postcode].iloc[0]
            st.success(
                f"📍 **{postcode_data['postcode']} - {postcode_data['city']}**: "
                f"{postcode_data['total_crimes']} crimes per 1,000 population predicted "
                f"({postcode_data['risk_level'].upper()} RISK)"
            )
        
        # Create and display map
        crime_map = create_crime_map(df, selected_risk, selected_postcode)
        map_data = st_folium(crime_map, width=700, height=500)
        
        # Handle map clicks
        if map_data['last_object_clicked_popup']:
            clicked_popup = map_data['last_object_clicked_popup']
            # Extract postcode from popup (this is a simplified approach)
            # In a real implementation, you'd want a more robust way to handle this
            pass
    

    with col3:
    # Crime type totals across all areas
        crime_totals = {
            'Theft': df['theft'].sum(),
            'Violence': df['violence'].sum(),
            'Sexual': df['sexual'].sum(),
            'Other': df['other'].sum()
        }
        
        for crime_type, total in sorted(crime_totals.items(), key=lambda x: x[1], reverse=True):
            st.metric(crime_type, total)    
    
    # Footer
    st.divider()
    st.markdown("""
    <div style="text-align: center; color: #6b7280; font-size: 0.9em;">
        <p>⚠️ Crime predictions are based on AI analysis of historical data and current trends
        Results are estimates and should be used for informational purposes only.</p>
        <p>Data updated for prediction period: <strong>July and August 2025</strong></p>
    </div>
    """.format(dict(months)[selected_month]), unsafe_allow_html=True)

if __name__ == "__main__":
    main()