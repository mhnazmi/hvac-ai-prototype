import os
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title="AI HVAC Learning Platform", layout="wide")
st.title("⚡ AI-Enabled Intelligent HVAC Platform")
st.caption("Real-Time Telemetry | Intelligent Psychrometrics | AI Assistant & Diagnostics")

st.sidebar.header("🎛️ Live Sensor Controls")
dry_bulb = st.sidebar.slider("Dry Bulb Temp (°C)", 10.0, 45.0, 25.0, 0.5)
rel_humidity = st.sidebar.slider("Relative Humidity (%)", 10.0, 95.0, 60.0, 1.0)
airflow = st.sidebar.slider("Airflow Rate (m³/h)", 100, 1000, 500, 50)

st.subheader("🔍 Predictive Diagnostics & Anomaly Detection")
if rel_humidity > 85 and dry_bulb < 20:
    st.error("⚠️ **FAULT DETECTED:** High saturation risk at cooling coil.")
elif dry_bulb > 35:
    st.warning("⚠️ **WARNING:** High thermal load on Air Handling Unit (AHU).")
else:
    st.success("✅ **SYSTEM STATUS NORMAL:** Operating within safe parameters.")
st.divider()

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📊 Dynamic Psychrometric Plot")
    enthalpy = round(1.006 * dry_bulb + (rel_humidity / 100) * 15.0, 2)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[dry_bulb], y=[rel_humidity], mode='markers+text', 
        name='Current State', text=[f"{dry_bulb}°C, {rel_humidity}% RH"],
        textposition="top center", marker=dict(size=16, color='crimson')
    ))
    fig.update_layout(xaxis_title="Dry Bulb Temp (°C)", yaxis_title="Relative Humidity (%)",
                      xaxis=dict(range=[10, 45]), yaxis=dict(range=[10, 100]), height=380)
    st.plotly_chart(fig, width="stretch")
    st.metric(label="Calculated Air Enthalpy", value=f"{enthalpy} kJ/kg")

with col2:
    st.subheader("🤖 Context-Aware AI Lab Assistant")
    
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Hi! I'm your AI HVAC tutor. Ask me anything about this experiment!"}]

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input("Ask a question..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        # MOCK AI RESPONSE BASED ON LIVE DATA
        if "fault" in prompt.lower() or "warning" in prompt.lower():
            bot_reply = f"Looking at your live data ({dry_bulb}°C and {rel_humidity}% RH), the system is hitting a threshold. If humidity exceeds 85%, condensation risks increase exponentially."
        elif "enthalpy" in prompt.lower():
            bot_reply = f"Your current calculated enthalpy is {enthalpy} kJ/kg. This represents the total heat content of the air. To lower it, you need to decrease either temperature or humidity."
        else:
            bot_reply = f"That is a great question! Based on your current inputs (Temp: {dry_bulb}°C, RH: {rel_humidity}%, Airflow: {airflow} m³/h), the psychrometric state is plotting normally. If you lower the temperature below the dew point, you will see a phase change occur on the cooling coil."

        st.session_state.messages.append({"role": "assistant", "content": bot_reply})
        st.chat_message("assistant").write(bot_reply)
