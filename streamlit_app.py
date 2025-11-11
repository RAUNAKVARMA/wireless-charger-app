import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime

if "did_redirect" not in st.session_state:
    st.session_state.did_redirect = True
    # st.switch_page("pages/YourPage.py")

st.set_page_config(
    page_title="WPT Efficiency Predictor",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS ---
st.markdown("""
<style>
    .metric-card {background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 20px; border-radius: 10px; color: white; margin: 10px 0;}
    .prediction-result {background: #f0f2f6; padding: 20px; border-radius: 10px; border-left: 4px solid #667eea;}
</style>
""", unsafe_allow_html=True)

# --- Title ---
st.title("⚡ Wireless Power Transfer Efficiency Predictor")
st.markdown(
    "*Physics-based predictions using real EV hardware equations ([see full publication](https://drive.google.com/file/d/16QYTlp2hGZ_Bwb6hl85dQLVggLr8U5Js/view?usp=drivesdk))*"
)


# --- Physics-based prediction function (from your paper) ---
def predict_efficiency(
    gap_mm, offset_mm, freq_khz, tx_temp, rx_temp, q_factor, current, voltage,
    ferrite, shield, oscillation_mode=None, coil_geometry=None
):
    gap_cm = gap_mm / 10.0
    offset_cm = offset_mm / 10.0
    L = 60e-6  # prototype value
    # Map descriptive labels to A/B config internally
    ferrite_map = {
        "A1 (Ferrite, 200x0.12mm Litz)": "A",
        "A2 (Ferrite, 200x0.12mm Litz)": "A",
        "B1 (No ferrite, 100x0.20mm Litz)": "B",
        "B2 (No ferrite, 100x0.20mm Litz)": "B"
    }
    ferrite_code = ferrite_map.get(ferrite, "A")
    R_dict = {'A': 0.055, 'B': 0.030}
    R = R_dict.get(ferrite_code, 0.055)
    freq = freq_khz * 1e3
    omega = 2 * np.pi * freq
    q_adj = q_factor / 150
    eta = 1 - (R/(L*omega*q_adj))
    if gap_cm > 15:
        eta *= (0.96 - 0.007*(gap_cm - 15))
    if offset_cm == 5:
        eta *= 0.984
    elif offset_cm == 10:
        eta *= 0.95
    elif offset_cm == 15:
        eta *= 0.905
    if oscillation_mode == 'mechanical':
        eta *= 1.0115
    elif oscillation_mode == 'electrical':
        eta *= 1.025
    geometry_efficiency = {
        "Circle (85–95%)": 0.95,
        "Elliptical (70–75%)": 0.73,
        "Circular Rectangular (90%)": 0.90,
        "Multi-Threaded (65–70%)": 0.68,
        "Square (85–90%)": 0.89,
        "Rectangular (75–90%)": 0.85,
        "Triangular (-)": 0.83,
        "Cross-Shaped (>90%)": 0.92,
        "Hexagonal (>90%)": 0.92,
        "Octagonal (>85%)": 0.88,
        "Pentagonal (Medium, %)": 0.86,
        "X-Pad (90–95%)": 0.93,
        "Segmented (90–95%)": 0.93,
        "Flux Pipe (>80%)": 0.81,
        "Homogeneous (-)": 0.80
    }
    shape_mult = geometry_efficiency.get(coil_geometry, 0.85)
    eta *= shape_mult
    eta = max(min(eta, 0.98), 0.65)  # expanded lower limit for poor geometries
    return eta*100.0

def predict_power(voltage, current, efficiency):
    return voltage * current * (efficiency / 100)

# --- Sidebar Navigation ---
st.sidebar.title("📍 Navigation")
page = st.sidebar.radio("Select Option:", ["🔮 Single Prediction", "📊 Frequency Optimization", "📈 Batch Analysis", "ℹ️ About"])

# ========== SINGLE PREDICTION ===========
if page == "🔮 Single Prediction":
    st.header("Single Configuration Prediction")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🔧 Physical Parameters")
        gap_mm = st.slider("Gap Distance (mm)", 100.0, 250.0, 150.0, 5.0)
        offset_mm = st.slider("Lateral Offset (mm)", 0.0, 150.0, 0.0, 5.0)
        frequency_khz = st.slider("Operating Frequency (kHz)", 80, 100, 85, 1)
    with col2:
        st.subheader("🌡️ Thermal & Electrical")
        tx_temp_c = st.slider("Tx Temperature (°C)", 30, 60, 45, 1)
        rx_temp_c = st.slider("Rx Temperature (°C)", 25, 55, 43, 1)
        q_factor = st.slider("Q-Factor (prototype: 130–220)", 130, 220, 170, 5)
    col3, col4 = st.columns(2)
    with col3:
        st.subheader("⚙️ Configuration")
        input_current_a = st.slider("Input Current (A)", 2.0, 10.0, 5.0, 0.1)
        input_voltage_v = st.slider("Input Voltage (V)", 200, 400, 350, 5)
    with col4:
        st.subheader("🛠️ Hardware")
        ferrite_config = st.selectbox(
            "Ferrite Config",
            [
                "A1 (Ferrite, 200x0.12mm Litz)",
                "A2 (Ferrite, 200x0.12mm Litz)",
                "B1 (No ferrite, 100x0.20mm Litz)",
                "B2 (No ferrite, 100x0.20mm Litz)"
            ]
        )
        shield_type = st.selectbox("Shield Type", ["none", "aluminum", "mu-metal", "ferrite"])
        oscillation_mode = st.selectbox("Oscillation/Modulation Mode", ["none", "mechanical", "electrical"])
        coil_geometry = st.selectbox(
            "Coil Geometry",
            [
                "Circle (85–95%)", "Elliptical (70–75%)", "Circular Rectangular (90%)",
                "Multi-Threaded (65–70%)", "Square (85–90%)", "Rectangular (75–90%)", "Triangular (-)",
                "Cross-Shaped (>90%)", "Hexagonal (>90%)", "Octagonal (>85%)", "Pentagonal (Medium, %)",
                "X-Pad (90–95%)", "Segmented (90–95%)", "Flux Pipe (>80%)", "Homogeneous (-)"
            ]
        )
    if st.button("🎯 Predict", use_container_width=True, type="primary"):
        efficiency = predict_efficiency(
            gap_mm, offset_mm, frequency_khz, tx_temp_c, rx_temp_c, q_factor, input_current_a, input_voltage_v,
            ferrite_config, shield_type, oscillation_mode, coil_geometry
        )
        power = predict_power(input_voltage_v, input_current_a, efficiency)
        ci_lower = max(80, efficiency - 1.5)
        ci_upper = min(98, efficiency + 1.5)
        st.markdown("---")
        st.subheader("📊 Prediction Results")
        col_res1, col_res2, col_res3 = st.columns(3)
        with col_res1:
            st.metric("Efficiency", f"{efficiency:.1f}%", f"±1.5%", delta_color="off")
            st.caption(f"Range: {ci_lower:.1f}% - {ci_upper:.1f}%\n_Based on measured prototype_")
        with col_res2:
            st.metric("Delivered Power", f"{power:.1f}W", f"of {input_voltage_v * input_current_a:.1f}W input")
        with col_res3:
            loss_pct = 100 - efficiency
            st.metric("Power Loss", f"{loss_pct:.1f}%", f"{(input_voltage_v*input_current_a)*(loss_pct/100):.1f}W")
        fig = go.Figure()
        fig.add_trace(go.Indicator(mode="gauge+number", value=efficiency, title={'text': "Efficiency (%)"}, domain={'x': [0, 0.45], 'y': [0, 1]}, gauge={'axis': {'range': [80, 98]}, 'bar': {'color': "#667eea"}, 'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 97}}))
        fig.add_trace(go.Indicator(mode="gauge+number", value=power, title={'text': "Power (W)"}, domain={'x': [0.55, 1], 'y': [0, 1]}, gauge={'axis': {'range': [0, input_voltage_v * input_current_a]}, 'bar': {'color': "#764ba2"}}))
        fig.update_layout(height=400, margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig, use_container_width=True)
        st.subheader("📋 Configuration Summary")
        summary_data = {
            'Parameter': ['Gap', 'Offset', 'Frequency', 'Tx Temp', 'Rx Temp', 'Q-Factor', 'Input Current', 'Input Voltage', 'Ferrite', 'Shield', 'Oscillation', 'Coil Geometry'],
            'Value': [f'{gap_mm}mm', f'{offset_mm}mm', f'{frequency_khz}kHz', f'{tx_temp_c}°C', f'{rx_temp_c}°C', f'{q_factor}', f'{input_current_a}A', f'{input_voltage_v}V', ferrite_config, shield_type, oscillation_mode, coil_geometry]
        }
        st.dataframe(pd.DataFrame(summary_data), use_container_width=True)

# ========== FREQUENCY OPTIMIZATION ===========
elif page == "📊 Frequency Optimization":
    st.header("Optimize Operating Frequency (Physics Model)")
    col1, col2 = st.columns(2)
    with col1:
        gap_mm = st.slider("Gap Distance (mm)", 100.0, 250.0, 150.0, 5.0, key="freq_gap")
        offset_mm = st.slider("Lateral Offset (mm)", 0.0, 150.0, 0.0, 5.0, key="freq_offset")
        q_factor = st.slider("Q-Factor", 130, 220, 170, 5)
        ferrite_config = st.selectbox("Ferrite Config", ["A1 (Ferrite, 200x0.12mm Litz)", "A2 (Ferrite, 200x0.12mm Litz)", "B1 (No ferrite, 100x0.20mm Litz)", "B2 (No ferrite, 100x0.20mm Litz)"], key="freq_ferrite")
        shield_type = st.selectbox("Shield Type", ["none", "aluminum", "mu-metal", "ferrite"], key="freq_shield")
        oscillation_mode = st.selectbox("Oscillation", ["none", "mechanical", "electrical"], key="freq_osc")
        coil_geometry = st.selectbox(
            "Coil Geometry",
            [
                "Circle (85–95%)", "Elliptical (70–75%)", "Circular Rectangular (90%)",
                "Multi-Threaded (65–70%)", "Square (85–90%)", "Rectangular (75–90%)", "Triangular (-)",
                "Cross-Shaped (>90%)", "Hexagonal (>90%)", "Octagonal (>85%)", "Pentagonal (Medium, %)",
                "X-Pad (90–95%)", "Segmented (90–95%)", "Flux Pipe (>80%)", "Homogeneous (-)"
            ], key="freq_geom"
        )
    with col2:
        input_current_a = st.slider("Input Current (A)", 2.0, 10.0, 5.0, 0.1, key="freq_i")
        input_voltage_v = st.slider("Input Voltage (V)", 200, 400, 350, 5, key="freq_v")
    if st.button("🔍 Scan Frequencies", use_container_width=True, type="primary"):
        frequencies = np.arange(80, 101, 1)
        efficiencies = [predict_efficiency(gap_mm, offset_mm, freq, 45, 43, q_factor, input_current_a, input_voltage_v, ferrite_config, shield_type, oscillation_mode, coil_geometry) for freq in frequencies]
        powers = [predict_power(input_voltage_v, input_current_a, eff) for eff in efficiencies]
        idx_opt = np.argmax(efficiencies)
        optimal_freq = int(frequencies[idx_opt])
        optimal_eff = efficiencies[idx_opt]
        optimal_power = powers[idx_opt]
        st.markdown("---")
        col_opt1, col_opt2, col_opt3 = st.columns(3)
        with col_opt1: st.metric("Optimal Frequency", f"{optimal_freq}kHz", "")
        with col_opt2: st.metric("Max Efficiency", f"{optimal_eff:.1f}%", "")
        with col_opt3: st.metric("Max Power", f"{optimal_power:.1f}W", f"vs {input_voltage_v*input_current_a:.1f}W")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=frequencies, y=efficiencies, mode='lines+markers', name='Efficiency', line=dict(color='#667eea', width=3), marker=dict(size=6)))
        fig.add_vline(x=optimal_freq, line_dash="dash", line_color="green", annotation_text=f"Optimal: {optimal_freq}kHz", annotation_position="top")
        fig.update_layout(title="Efficiency vs Frequency", xaxis_title="Frequency (kHz)", yaxis_title="Efficiency (%)", hovermode='x unified', template='plotly_white', height=400)
        st.plotly_chart(fig, use_container_width=True)
        df_freq = pd.DataFrame({'Frequency (kHz)': frequencies, 'Efficiency (%)': [f'{e:.1f}' for e in efficiencies], 'Power (W)': [f'{p:.1f}' for p in powers]})
        st.dataframe(df_freq, use_container_width=True)

# ========== BATCH ANALYSIS ===========
elif page == "📈 Batch Analysis":
    st.header("Batch Prediction Analysis")
    st.info("Edit or add configurations below to see batch statistics (")
    sample_configs = pd.DataFrame({
        'gap_mm': [150.0, 180.0, 200.0, 230.0, 120.0],
        'offset_mm': [0.0, 5.0, 10.0, 15.0, 0.0],
        'frequency_khz': [85, 90, 95, 100, 80],
        'tx_temp_c': [45, 45, 45, 45, 45],
        'rx_temp_c': [43, 43, 43, 43, 43],
        'q_factor': [170, 150, 210, 180, 200],
        'input_current_a': [5.0, 4.0, 6.0, 8.0, 2.5],
        'input_voltage_v': [350, 250, 250, 300, 400],
        'ferrite_config': ["A1 (Ferrite, 200x0.12mm Litz)", "B1 (No ferrite, 100x0.20mm Litz)", "A2 (Ferrite, 200x0.12mm Litz)", "B2 (No ferrite, 100x0.20mm Litz)", "A1 (Ferrite, 200x0.12mm Litz)"],
        'shield_type': ['mu-metal', 'ferrite', 'none', 'mu-metal', 'aluminum'],
        'oscillation_mode': ['none', 'mechanical', 'electrical', 'mechanical', 'none'],
        'coil_geometry': [
            "Circle (85–95%)", "Elliptical (70–75%)", "Pancake (flat)", "Rectangular (75–90%)", "X-Pad (90–95%)"
        ]
    })
    edited_df = st.data_editor(sample_configs, use_container_width=True, num_rows="dynamic")
    if st.button("🚀 Analyze Batch", use_container_width=True, type="primary"):
        results = []
        for idx, row in edited_df.iterrows():
            eff = predict_efficiency(row['gap_mm'], row['offset_mm'], row['frequency_khz'], row['tx_temp_c'], row['rx_temp_c'], row['q_factor'], row['input_current_a'], row['input_voltage_v'], row['ferrite_config'], row['shield_type'], row['oscillation_mode'], row['coil_geometry'])
            pwr = predict_power(row['input_voltage_v'], row['input_current_a'], eff)
            results.append({
                'Config ID': idx + 1,
                'Gap (mm)': row['gap_mm'],
                'Offset (mm)': row['offset_mm'],
                'Frequency (kHz)': row['frequency_khz'],
                'Efficiency (%)': round(eff, 1),
                'Power (W)': round(pwr, 1),
                'Loss (%)': round(100-eff, 1),
                'Oscillation': row.get('oscillation_mode', 'none'),
                'Coil Geometry': row['coil_geometry']
            })
        results_df = pd.DataFrame(results)
        st.markdown("---")
        st.subheader("📊 Batch Results")
        st.dataframe(results_df, use_container_width=True)
        col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
        with col_stat1: st.metric("Avg Efficiency", f"{results_df['Efficiency (%)'].mean():.1f}%")
        with col_stat2: st.metric("Max Efficiency", f"{results_df['Efficiency (%)'].max():.1f}%")
        with col_stat3: st.metric("Min Efficiency", f"{results_df['Efficiency (%)'].min():.1f}%")
        with col_stat4: st.metric("Avg Power", f"{results_df['Power (W)'].mean():.1f}W")
        col_chart1, col_chart2 = st.columns(2)
        with col_chart1:
            fig_eff = go.Figure()
            fig_eff.add_trace(go.Bar(x=results_df['Config ID'], y=results_df['Efficiency (%)'], marker_color='#667eea', text=results_df['Efficiency (%)'], textposition='auto'))
            fig_eff.update_layout(title="Efficiency by Configuration", height=350)
            st.plotly_chart(fig_eff, use_container_width=True)
        with col_chart2:
            fig_pwr = go.Figure()
            fig_pwr.add_trace(go.Bar(x=results_df['Config ID'], y=results_df['Power (W)'], marker_color='#764ba2', text=results_df['Power (W)'], textposition='auto'))
            fig_pwr.update_layout(title="Power by Configuration", height=350)
            st.plotly_chart(fig_pwr, use_container_width=True)

# ========== ABOUT TAB ===========
elif page == "ℹ️ About":
    st.header("📘 About This Predictor")
    col_about1, col_about2 = st.columns(2)
    with col_about1:
        st.subheader("🎯 What is this?")
        st.markdown(r"""
**Physics-based EV charger predictor**  
Formula reflects:
- Real gap/offset/ferrite/oscillation/coil effects (see your paper Table 1, Table 5–6)
- Model:
    
    $$
    \eta = 1 - \frac{R}{L\omega Q_{\mathrm{rel}}} \times \text{Geometry Multiplier}
    $$
""")

    with col_about2:
        st.subheader("⚙️ Model Details")
        st.markdown(r"""
- Core parameters match: Q-factor, coil config, gap, offset, geometry
- Oscillation/freq modulation selectable
- All UI/table features preserved
""")
    st.markdown("---")
    st.subheader("🚀 Quick Start Guide")
    st.markdown(r"""
1. Go to a tab
2. Set your hardware/test config
3. Click Predict/Analyze
4. Results use published equations + geometry
""")

st.markdown("---")
st.markdown("<p style='text-align: center; color: gray;'>⚡ WPT Efficiency Predictor | Research prototype equations + geometry, all features and UI preserved</p>", unsafe_allow_html=True)

