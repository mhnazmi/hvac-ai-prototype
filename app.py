import os
import streamlit as st
import plotly.graph_objects as go

# 1. Page Config
st.set_page_config(page_title="AI HVAC Learning Platform", layout="wide")

st.title("⚡ AI-Enabled Intelligent HVAC Platform")
st.caption("Real-Time Telemetry | Intelligent Psychrometrics | AI Assistant & Diagnostics")

# 2. Sidebar - Simulated Sensor Telemetry
st.sidebar.header("🎛️ Live Sensor Controls")
st.sidebar.markdown("Simulate HVAC laboratory sensor inputs:")
dry_bulb = st.sidebar.slider("Dry Bulb Temp (°C)", 10.0, 45.0, 25.0, 0.5)
rel_humidity = st.sidebar.slider("Relative Humidity (%)", 10.0, 95.0, 60.0, 1.0)
airflow = st.sidebar.slider("Airflow Rate (m³/h)", 100, 1000, 500, 50)

# 3. Real-Time Predictive Diagnostics
st.subheader("🔍 Predictive Diagnostics & Anomaly Detection")
if rel_humidity > 85 and dry_bulb < 20:
    st.error("⚠️ **FAULT DETECTED:** High saturation risk at cooling coil. Potential condensate buildup or coil valve failure!")
elif dry_bulb > 35:
    st.warning("⚠️ **WARNING:** High thermal load on Air Handling Unit (AHU). Sub-optimal cooling efficiency.")
else:
    st.success("✅ **SYSTEM STATUS NORMAL:** Operating within safe psychrometric parameters.")

st.divider()

col1, col2 = st.columns([1, 1])

# 4. Work Package 3: Dynamic Psychrometric Visualization
with col1:
    st.subheader("📊 Dynamic Psychrometric Plot")
    
    # Calculate simple enthalpy approximation for display
    enthalpy = round(1.006 * dry_bulb + (rel_humidity / 100) * 15.0, 2)
    
    # Create interactive chart with Plotly
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[dry_bulb], 
        y=[rel_humidity], 
        mode='markers+text', 
        name='Current State',
        text=[f"{dry_bulb}°C, {rel_humidity}% RH"],
        textposition="top center",
        marker=dict(size=16, color='crimson')
    ))
    
    fig.update_layout(
        xaxis_title="Dry Bulb Temperature (°C)",
        yaxis_title="Relative Humidity (%)",
        xaxis=dict(range=[10, 45]),
        yaxis=dict(range=[10, 100]),
        height=380,
        margin=dict(l=20, r=20, t=30, b=20)
    )
    
    st.plotly_chart(fig, width="stretch")
    st.metric(label="Calculated Air Enthalpy", value=f"{enthalpy} kJ/kg")

# 5. Work Package 2: Context-Aware AI Tutor
with col2:
    st.subheader("🤖 Context-Aware AI Lab Assistant")
    
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hi! I'm your AI HVAC tutor trained on your lab manual. Ask me anything about this experiment or why your sensors are showing these readings!"}
        ]

    # Render chat history
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    # Handle chat input
    if prompt := st.chat_input("Ask a question (e.g., 'Explain the process line for this reading')..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        # Connect to Gemini API if key exists, otherwise print exact error
        api_key = st.secrets.get("GEMINI_API_KEY", None)
        
        if api_key:
            try:
                from google import genai
                client = genai.Client(api_key=api_key)
                system_instruction = f"You are an AI HVAC tutor in a university lab. The current sensor readings are: Dry Bulb Temp = {dry_bulb}°C, Relative Humidity = {rel_humidity}%, Airflow = {airflow} m³/h. Use this context to answer concisely."
                
                response = client.models.generate_content(
                    model='gemini-2.5-flash',
                    contents=f"{system_instruction}\nUser question: {prompt}"
                )
                bot_reply = response.text
            except Exception as e:
                bot_reply = f"*(API Error)*: {str(e)}"
        else:
            bot_reply = f"*(No API Key Found)*: `GEMINI_API_KEY` was not detected in Streamlit Secrets."

        st.session_state.messages.append({"role": "assistant", "content": bot_reply})
        st.chat_message("assistant").write(bot_reply)
