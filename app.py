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
        # COMPREHENSIVE MOCK AI RESPONSE ENGINE
        user_input = prompt.lower()
        
        # 1. Greetings & Identity
        if any(word in user_input for word in ["hello", "hi", "hey", "who are you", "what are you"]):
            bot_reply = "Hello! I am your AI HVAC Lab Assistant. I am here to help you understand psychrometric processes, analyze live data, and diagnose system operations without waiting for a professor!"
            
        # 2. Platform / Purpose
        elif any(word in user_input for word in ["website", "platform", "this app", "what is this", "hackathon"]):
            bot_reply = "This is an AI-enabled Intelligent HVAC Learning Platform. It visualizes real-time psychrometric data and allows you to experiment with HVAC parameters while providing instant, context-aware guidance."
            
        # 3. Telemetry: Temperature
        elif any(word in user_input for word in ["temperature", "dry bulb", "hot", "cold"]):
            bot_reply = f"The current dry bulb temperature is reading {dry_bulb}°C. This is the ambient thermodynamic temperature of the air entering the system."
            
        # 4. Telemetry: Humidity
        elif any(word in user_input for word in ["humidity", "relative humidity", "rh", "moisture"]):
            bot_reply = f"The relative humidity is currently {rel_humidity}%. This means the air is holding {rel_humidity}% of the maximum moisture it can contain at {dry_bulb}°C before saturation occurs."
            
        # 5. Telemetry: Airflow
        elif any(word in user_input for word in ["airflow", "velocity", "cfm", "m3/h", "flow"]):
            bot_reply = f"The blower is currently maintaining an airflow rate of {airflow} m³/h. Proper airflow is critical to ensuring effective heat transfer across the cooling and heating coils."
            
        # 6. Psychrometrics: Enthalpy
        elif any(word in user_input for word in ["enthalpy", "total heat", "energy"]):
            bot_reply = f"Based on your inputs, the calculated enthalpy is approximately {enthalpy} kJ/kg. Enthalpy represents the total heat energy in the air, combining both sensible heat (temperature) and latent heat (moisture)."
            
        # 7. Psychrometrics: Dew Point & Condensation
        elif any(word in user_input for word in ["dew point", "condensation", "condense", "water phase", "dew"]):
            # Rule of thumb calculation for realistic mock data
            dp_approx = round(dry_bulb - ((100 - rel_humidity) / 5), 1) 
            bot_reply = f"At {dry_bulb}°C and {rel_humidity}% RH, the approximate dew point is {dp_approx}°C. If the cooling coil surface temperature drops below {dp_approx}°C, moisture in the air will begin to condense into liquid water."
            
        # 8. Components: Cooling Coil
        elif any(word in user_input for word in ["cooling coil", "chilled water", "chiller"]):
            bot_reply = "The cooling coil lowers the air temperature by transferring heat to chilled water. If the coil cools the air below its dew point, it will also dehumidify the air by causing condensation."
            
        # 9. Components: Heating Unit
        elif any(word in user_input for word in ["heating", "heater", "reheat"]):
            bot_reply = "The heating unit adds sensible heat to the air. As the dry bulb temperature rises, the relative humidity drops, even though the absolute moisture content (humidity ratio) remains the same."
            
        # 10. Components: Humidifier
        elif "humidifier" in user_input:
            bot_reply = "The humidifier injects moisture (steam or atomized water) directly into the airstream. This increases the latent heat and relative humidity without necessarily changing the sensible temperature."
            
        # 11. Components: AHU
        elif "ahu" in user_input or "handling unit" in user_input:
            bot_reply = "The Air Handling Unit (AHU) is the large metal enclosure containing the blower, coils, filters, and dampers. It is the heart of the HVAC system where all air conditioning processes physically take place."
            
        # 12. Faults & Diagnostics: Dynamic Warnings
        elif any(word in user_input for word in ["fault", "warning", "danger", "alarm", "predictive", "diagnostic"]):
            if rel_humidity > 85:
                bot_reply = f"⚠️ DIAGNOSTIC ALERT: At {rel_humidity}% RH, you are dangerously close to full saturation. This indicates a potential failure in the dehumidification process or a cooling coil valve malfunction."
            elif dry_bulb > 35:
                bot_reply = f"⚠️ DIAGNOSTIC ALERT: The intake temperature of {dry_bulb}°C is placing an extreme thermal load on the AHU. Cooling efficiency will be significantly reduced at this state."
            else:
                bot_reply = f"✅ SYSTEM NORMAL: The current state ({dry_bulb}°C, {rel_humidity}% RH) is within safe operational limits. No faults are currently detected in the AHU telemetry."

        # 13. Professor Alternative / Learning
        elif any(word in user_input for word in ["professor", "lecturer", "teach", "help me", "don't understand", "lost"]):
            bot_reply = "I am designed to be your instant lab partner! Instead of waiting for your professor during a busy lab, you can ask me to explain any psychrometric process or HVAC component right here."

        # 14. Catch-all / Fallback Engine
        else:
            bot_reply = f"That is a great question to explore in the lab! Based on your live data (Temp: {dry_bulb}°C, RH: {rel_humidity}%, Airflow: {airflow} m³/h), how do you think adjusting the heating or cooling coils would affect this process? Check the psychrometric chart on the left to see the current state point!"

        st.session_state.messages.append({"role": "assistant", "content": bot_reply})
        st.chat_message("assistant").write(bot_reply)
