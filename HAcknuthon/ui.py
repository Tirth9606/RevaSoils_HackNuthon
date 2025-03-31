import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
import time
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
PREDICTIONS_FILE = "soil_predictions.csv"  # Must match model.py

# Set page config
st.set_page_config(
    page_title="Soil Analysis Dashboard",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    /* Main page styling */
    .main {
        background-color: #f5f9f5;
        padding: 2rem;
    }
    
    /* Header styling */
    h1, h2, h3 {
        color: #2e7d32;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    /* Button styling */
    .stButton>button {
        background-color: #388e3c;
        color: white;
        border-radius: 8px;
        padding: 0.5rem 1.5rem;
        font-weight: 500;
        border: none;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        background-color: #2e7d32;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
    }
    
    /* Metric cards */
    .metric-card {
        background-color: white;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        margin-bottom: 1.5rem;
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
    }
    
    /* Recommendation cards */
    .recommendation-card {
        background-color: #f0fff4;
        border-left: 5px solid #4caf50;
        padding: 1.2rem;
        margin-bottom: 1rem;
        border-radius: 0 10px 10px 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    
    /* Tab styling */
    .stTabs [role="tablist"] {
        margin-bottom: 1.5rem;
    }
    
    .stTabs [role="tab"] {
        padding: 0.8rem 1.5rem;
        border-radius: 8px 8px 0 0;
        font-weight: 500;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #e8f5e9;
        color: #2e7d32;
        border-bottom: 3px solid #4caf50;
    }
    
    /* Graph container */
    .stPlotlyChart {
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        background-color: white;
        padding: 1rem;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #e8f5e9;
        padding: 2rem 1rem;
    }
    
    /* Error messages */
    .stAlert {
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# -------------------
# Data Processing Functions
# -------------------
def preprocess_data(df):
    if 'Records' in df.columns:
        df['Soil_ID'] = df['Records'].str.split('-').str[0]
        df['Moisture_Level'] = df['Soil_ID'].str.split('_').str[1]
        df = df.drop(columns=['Records'])
    
    # Aggregate multiple measurements per Soil_ID using median
    aggregated = df.groupby('Soil_ID').median(numeric_only=True).reset_index()
    
    sensor_cols = ['Moist', 'EC (u/10 gram)', 'Ph', 
                 'Nitro (mg/10 g)', 'Posh Nitro (mg/10 g)', 'Pota Nitro (mg/10 g)']
    
    existing_sensor_cols = [col for col in sensor_cols if col in aggregated.columns]
    aggregated[existing_sensor_cols] = aggregated[existing_sensor_cols].replace(0, np.nan)
    
    if existing_sensor_cols:
        imputer = KNNImputer(n_neighbors=3)
        aggregated[existing_sensor_cols] = imputer.fit_transform(aggregated[existing_sensor_cols])
    
    return aggregated

def process_spectral_data(df):
    wavelengths = ['410', '435', '460', '485', '510', '535', 
                 '560', '585', '610', '645', '680', '705', 
                 '730', '760', '810', '860', '900', '940']
    
    existing_wavelengths = [w for w in wavelengths if w in df.columns]
    
    if existing_wavelengths:
        df[existing_wavelengths] = savgol_filter(df[existing_wavelengths], 
                                               window_length=11, 
                                               polyorder=2,
                                               axis=1)
        
        scaler = StandardScaler()
        spectral_data = scaler.fit_transform(df[existing_wavelengths])
        
        deriv = np.gradient(spectral_data, axis=1)
        
        if '860' in df.columns and '645' in df.columns:
            df['NDI'] = (df['860'] - df['645']) / (df['860'] + df['645'])
        if '730' in df.columns and '680' in df.columns:
            df['SIR'] = df['730'] / df['680']
        
        deriv_cols = [f'd{w}' for w in existing_wavelengths]
        df = pd.concat([df, pd.DataFrame(deriv, columns=deriv_cols)], axis=1)
    
    return df

def create_features(df):
    existing_cols = df.columns.tolist()
    
    vis_cols = [f'{w}' for w in range(400, 700, 10) if f'{w}' in existing_cols]
    nir_cols = [f'{w}' for w in range(700, 1000, 10) if f'{w}' in existing_cols]
    
    if vis_cols:
        df['VIS_avg'] = df[vis_cols].mean(axis=1)
    if nir_cols:
        df['NIR_avg'] = df[nir_cols].mean(axis=1)
    
    wavelengths = [str(w) for w in range(400, 1000, 10) if str(w) in existing_cols]
    if wavelengths:
        pca = PCA(n_components=min(5, len(wavelengths)))
        pca_features = pca.fit_transform(df[wavelengths])
        df_pca = pd.DataFrame(pca_features, columns=[f'PCA_{i+1}' for i in range(pca_features.shape[1])])
        df = pd.concat([df, df_pca], axis=1)
    
    return df

def train_model(X, y):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

def get_final_predictions(df):
    """Process data to get one value per sensor per Soil_ID"""
    # Group by Soil_ID and calculate mean for each sensor
    sensor_cols = ['Moist', 'EC (u/10 gram)', 'Ph', 
                 'Nitro (mg/10 g)', 'Posh Nitro (mg/10 g)', 'Pota Nitro (mg/10 g)']
    
    # Only use columns that exist in the data
    existing_sensor_cols = [col for col in sensor_cols if col in df.columns]
    
    if not existing_sensor_cols:
        return None
    
    # Group by Soil_ID and calculate median (more robust than mean)
    aggregated = df.groupby('Soil_ID')[existing_sensor_cols].median().reset_index()
    
    # If we have moisture level information, we can use it to enhance predictions
    if 'Moisture_Level' in df.columns:
        # Create a mapping from moisture level to numeric value
        moisture_mapping = {'0ml': 0, '25ml': 25, '50ml': 50}
        df['Moisture_Value'] = df['Moisture_Level'].map(moisture_mapping)
        
        # For each Soil_ID, get the moisture level trend
        moisture_trend = df.groupby('Soil_ID')['Moisture_Value'].first().reset_index()
        aggregated = pd.merge(aggregated, moisture_trend, on='Soil_ID')
    
    return aggregated

# -------------------
# UI Components
# -------------------
def show_upload_section():
    st.sidebar.header("Upload Soil Data")
    uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state['raw_data'] = df
            return df
        except Exception as e:
            st.sidebar.error(f"Error reading file: {str(e)}")
            return None
    return None

def display_soil_health_summary(predictions):
    st.subheader("Soil Health Summary")
    
    # Create metrics row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown('<div class="metric-card">'
                   f'<h3>Moisture</h3>'
                   f'<h2>{predictions["Moist"].iloc[0]:.1f}</h2>'
                   '<p>Optimal range: 20-60</p>'
                   '</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">'
                   f'<h3>pH Level</h3>'
                   f'<h2>{predictions["Ph"].iloc[0]:.1f}</h2>'
                   '<p>Optimal range: 6.0-7.0</p>'
                   '</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">'
                   f'<h3>Nitrogen</h3>'
                   f'<h2>{predictions["Nitro (mg/10 g)"].iloc[0]:.1f} mg</h2>'
                   '<p>Optimal: >20 mg</p>'
                   '</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="metric-card">'
                   f'<h3>EC</h3>'
                   f'<h2>{predictions["EC (u/10 gram)"].iloc[0]:.1f} uS/cm</h2>'
                   '<p>Normal range: 100-1500</p>'
                   '</div>', unsafe_allow_html=True)
    
    # Soil health gauge
    health_score = calculate_health_score(predictions.iloc[0])
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = health_score,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Soil Health Score"},
        gauge = {
            'axis': {'range': [None, 100]},
            'steps': [
                {'range': [0, 40], 'color': "red"},
                {'range': [40, 70], 'color': "orange"},
                {'range': [70, 100], 'color': "green"}],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': health_score}
        }
    ))
    st.plotly_chart(fig, use_container_width=True)

def calculate_health_score(row):
    """Calculate a composite soil health score (0-100)"""
    # Normalize each parameter to 0-1 scale
    moisture_score = min(max((row['Moist'] - 10) / 50, 0), 1)  # 10-60 is reasonable range
    ph_score = 1 - abs(row['Ph'] - 6.5) / 3.5  # 6.5 is ideal, 0-14 range
    nitro_score = min(row['Nitro (mg/10 g)'] / 50, 1)  # Up to 50 is good
    ec_score = 1 - min(max(abs(row['EC (u/10 gram)'] - 800) / 800, 0), 1)  # 800 is ideal
    
    # Weighted average
    return (moisture_score * 0.3 + ph_score * 0.3 + nitro_score * 0.2 + ec_score * 0.2) * 100

def display_soil_parameters(final_df, predictions):
    st.subheader("Detailed Soil Parameters")
    
    # Ensure we have valid data
    if final_df.empty or predictions.empty:
        st.warning("No data available for visualization")
        return
    
    # Create safe identifier column if missing
    if 'Soil_ID' not in final_df.columns:
        final_df['Soil_ID'] = [f"Sample_{i}" for i in range(1, len(final_df)+1)]
    
    if 'Soil_ID' not in predictions.columns:
        predictions['Soil_ID'] = final_df['Soil_ID']
    
    # Create two columns for charts
    col1, col2 = st.columns(2)
    
    with col1:
        try:
            # Moisture and pH line chart
            if 'Moist' in predictions.columns and 'Ph' in predictions.columns:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=final_df['Soil_ID'],
                    y=predictions['Moist'],
                    name='Moisture',
                    line=dict(color='#2e7d32', width=2),
                    marker=dict(size=8)
                ))
                fig.add_trace(go.Scatter(
                    x=final_df['Soil_ID'],
                    y=predictions['Ph'] * 10,  # Scale for visibility
                    name='pH (scaled x10)',
                    line=dict(color='#0288d1', width=2),
                    marker=dict(size=8)
                ))
                fig.update_layout(
                    title='Moisture and pH Levels',
                    yaxis_title='Value',
                    xaxis_title='Sample ID',
                    hovermode='x unified',
                    plot_bgcolor='white',
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Moisture or pH data not available")
        except Exception as e:
            st.error(f"Error displaying moisture/pH chart: {str(e)}")
    
    with col2:
        try:
            # EC chart
            if 'EC (u/10 gram)' in predictions.columns:
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=final_df['Soil_ID'],
                    y=predictions['EC (u/10 gram)'],
                    name='EC',
                    marker_color='#ff9800'
                ))
                fig.update_layout(
                    title='Electrical Conductivity',
                    yaxis_title='uS/cm',
                    xaxis_title='Sample ID',
                    plot_bgcolor='white',
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("EC data not available")
        except Exception as e:
            st.error(f"Error displaying EC chart: {str(e)}")
    
    # Nutrient radar chart
    st.subheader("Nutrient Balance")
    try:
        nutrient_cols = ['Nitro (mg/10 g)', 'Posh Nitro (mg/10 g)', 'Pota Nitro (mg/10 g)']
        available_nutrients = [col for col in nutrient_cols if col in predictions.columns]
        
        if len(available_nutrients) >= 2:  # Need at least 2 nutrients for radar chart
            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(
                r=[predictions[col].mean() for col in available_nutrients],
                theta=available_nutrients,
                fill='toself',
                name='Nutrient Levels',
                fillcolor='rgba(76, 175, 80, 0.5)',
                line=dict(color='#2e7d32')
            ))
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, max(predictions[col].max() for col in available_nutrients) * 1.1]
                    )
                ),
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Insufficient nutrient data for radar chart")
    except Exception as e:
        st.error(f"Error displaying nutrient radar chart: {str(e)}")

def display_recommendations(predictions):
    st.subheader("Personalized Recommendations")
    
    # Get the first row of predictions (assuming we want recommendations for the first sample)
    pred = predictions.iloc[0]
    
    # Moisture recommendation
    moisture = pred['Moist']
    if moisture < 20:
        st.markdown('<div class="recommendation-card">'
                   '🚰 <strong>Irrigation Needed</strong>: Soil moisture is very low (below 20). '
                   'Increase watering frequency or amount.'
                   '</div>', unsafe_allow_html=True)
    elif moisture > 60:
        st.markdown('<div class="recommendation-card">'
                   '💧 <strong>Reduce Irrigation</strong>: Soil is oversaturated (above 60). '
                   'Reduce watering to prevent root rot and nutrient leaching.'
                   '</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="recommendation-card">'
                   '✅ <strong>Moisture Optimal</strong>: Current watering schedule appears appropriate.'
                   '</div>', unsafe_allow_html=True)
    
    # pH recommendation
    ph = pred['Ph']
    if ph < 6:
        st.markdown('<div class="recommendation-card">'
                   '🔼 <strong>Increase pH</strong>: Soil is too acidic (pH {:.1f}). '
                   'Add agricultural lime to raise pH to optimal range (6-7).'
                   '</div>'.format(ph), unsafe_allow_html=True)
    elif ph > 7:
        st.markdown('<div class="recommendation-card">'
                   '🔽 <strong>Lower pH</strong>: Soil is too alkaline (pH {:.1f}). '
                   'Add sulfur or organic matter to reduce pH.'
                   '</div>'.format(ph), unsafe_allow_html=True)
    else:
        st.markdown('<div class="recommendation-card">'
                   '✅ <strong>pH Optimal</strong>: Soil pH is in the ideal range for most crops.'
                   '</div>', unsafe_allow_html=True)
    
    # Nitrogen recommendation
    nitro = pred['Nitro (mg/10 g)']
    if nitro < 20:
        st.markdown('<div class="recommendation-card">'
                   '🌿 <strong>Nitrogen Deficient</strong>: Level is low ({:.1f} mg). '
                   'Apply nitrogen-rich fertilizer (e.g., urea, ammonium nitrate, or compost).'
                   '</div>'.format(nitro), unsafe_allow_html=True)
    elif nitro > 50:
        st.markdown('<div class="recommendation-card">'
                   '⚠️ <strong>Excess Nitrogen</strong>: Level is high ({:.1f} mg). '
                   'Reduce nitrogen inputs to prevent nutrient imbalance and potential plant damage.'
                   '</div>'.format(nitro), unsafe_allow_html=True)
    else:
        st.markdown('<div class="recommendation-card">'
                   '✅ <strong>Nitrogen Adequate</strong>: Levels are sufficient for plant growth.'
                   '</div>', unsafe_allow_html=True)
    
    # EC recommendation
    ec = pred['EC (u/10 gram)']
    if ec < 100:
        st.markdown('<div class="recommendation-card">'
                   '🧂 <strong>Low Salinity</strong>: EC is very low ({:.1f} uS/cm). '
                   'Consider adding balanced nutrients to irrigation water.'
                   '</div>'.format(ec), unsafe_allow_html=True)
    elif ec > 1500:
        st.markdown('<div class="recommendation-card">'
                   '⚠️ <strong>High Salinity</strong>: EC is very high ({:.1f} uS/cm). '
                   'Leach soil with clean water to reduce salt levels.'
                   '</div>'.format(ec), unsafe_allow_html=True)
    else:
        st.markdown('<div class="recommendation-card">'
                   '✅ <strong>EC Normal</strong>: Electrical conductivity is within optimal range.'
                   '</div>', unsafe_allow_html=True)
    
    # Fertilizer recommendation
    st.markdown("---")
    st.subheader("Suggested Fertilizer Blend")
    
    # Calculate ideal NPK ratio based on predictions
    N = max(0, 25 - pred['Nitro (mg/10 g)']) / 25 * 100
    P = max(0, 15 - pred['Posh Nitro (mg/10 g)']) / 15 * 100
    K = max(0, 30 - pred['Pota Nitro (mg/10 g)']) / 30 * 100
    
    # Ensure at least some minimum recommendation
    N = max(N, 10)
    P = max(P, 10)
    K = max(K, 10)
    
    fig = go.Figure(go.Bar(
        x=['Nitrogen (N)', 'Phosphorus (P)', 'Potassium (K)'],
        y=[N, P, K],
        marker_color=['#4CAF50', '#FFC107', '#F44336']
    ))
    fig.update_layout(
        title="Recommended Fertilizer Composition (%)",
        yaxis_title="Percentage of Nutrient Requirement",
        yaxis_range=[0, 100]
    )
    st.plotly_chart(fig, use_container_width=True)

def display_results(final_df, predictions):
    st.header("Soil Analysis Results")
    
    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs(["Summary", "Detailed Analysis", "Recommendations"])
    
    with tab1:
        display_soil_health_summary(predictions)
    
    with tab2:
        display_soil_parameters(final_df, predictions)
    
    with tab3:
        display_recommendations(predictions)

# -------------------
# Main App
# -------------------
def main():
    st.title("🌱 Smart Soil Analysis Dashboard")
    st.markdown("Upload your soil sensor data to get detailed analysis and farming recommendations")
    
    # Check for existing predictions
    if os.path.exists(PREDICTIONS_FILE):
        st.sidebar.success("Pre-computed predictions found!")
        use_existing = st.sidebar.checkbox("Use existing predictions", value=True)
    else:
        use_existing = False
    
    # File upload section
    df = show_upload_section()
    
    if not use_existing and df is not None:
        if st.sidebar.button("Analyze Soil Data"):
            with st.spinner("Processing soil data..."):
                try:
                    # Process data
                    processed_df = preprocess_data(df)
                    spectral_df = process_spectral_data(processed_df)
                    final_df = create_features(spectral_df)
                    
                    # Get final predictions
                    predictions = get_final_predictions(final_df)
                    
                    if predictions is None:
                        st.error("No valid sensor data found in the uploaded file.")
                        return
                    
                    st.session_state['final_df'] = final_df
                    st.session_state['predictions'] = predictions
                    st.session_state['processed'] = True
                    st.success("Analysis complete! View results below.")
                except Exception as e:
                    st.error(f"Error processing data: {str(e)}")
    
    # Load existing predictions if available
    if use_existing and os.path.exists(PREDICTIONS_FILE):
        try:
            predictions = pd.read_csv(PREDICTIONS_FILE)
            st.session_state['predictions'] = predictions
            st.session_state['processed'] = True
            st.success("Loaded pre-computed predictions!")
        except Exception as e:
            st.error(f"Error loading predictions: {str(e)}")
    
    if st.session_state.get('processed', False):
        display_results(
            st.session_state.get('final_df', pd.DataFrame()), 
            st.session_state['predictions']
        )

if __name__ == "__main__":
    main()