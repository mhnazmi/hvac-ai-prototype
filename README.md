# AI-Enabled Intelligent HVAC Learning Platform

A digital twin of the SIT HVAC teaching rig. One simulation core feeds all three
work packages, so the chart, the 3D-style schematic and the AI tutor always agree.

## Architecture

```
        Sensor + plant controls          Fault injection
                     |                          |
                     v                          v
        +------------------------------------------------+
        |   SIMULATION CORE  -  AHU digital twin          |
        |   PsychroLib (ASHRAE Fundamentals 2017)         |
        |   intake -> cooling coil -> reheat -> humidify  |
        +------------------------------------------------+
             |                  |                  |
             v                  v                  v
      WP3 Psychrometric    WP1 Immersive      WP2 AI Learning
        Visualization       Environment          Assistant
```

## Engineering model

- **Psychrometrics**: all properties (humidity ratio, enthalpy, dew point, wet
  bulb, specific volume) from PsychroLib, an implementation of the ASHRAE
  Handbook of Fundamentals. No hand-rolled correlations.
- **Cooling coil**: apparatus dew point / bypass factor model.
  `T_out = ADP + BF*(T_in - ADP)`, likewise for humidity ratio. Falls back to
  pure sensible cooling when the ADP sits above the intake dew point.
- **Reheat**: sensible only, `dT = Q / (m_dot * cp_moist)`, humidity ratio held.
- **Humidifier**: latent only, `dW = m_water / m_dot`, dry bulb held.
- **Loads**: `Q_total = m_dot * dh`, `Q_sens = m_dot * cp_moist * dT`,
  `Q_lat = Q_total - Q_sens`, `SHR = Q_sens / Q_total`.
- Energy balance closes to within 1e-4 W across the tested envelope.

## Fault injection

| Fault | Physical effect |
|---|---|
| Fouled cooling coil | bypass factor 0.12 -> 0.38, approach +3 K |
| Clogged air filter | airflow x 0.55 |
| Fan belt slipping | airflow x 0.70 |
| Low chilled water flow | approach +5 K |

The diagnostics engine compares the twin against clean-plant expectations and
raises findings, satisfying "compare normal vs faulty operation" (WP1),
"identification of faults" (WP2) and "detect abnormal operating conditions"
(WP3) from one mechanism.

## Running

```bash
pip install -r requirements.txt
streamlit run app.py
```

The AI tutor needs a Gemini API key. Either set an environment variable:

```bash
export GEMINI_API_KEY="your-key"
```

or add it to `.streamlit/secrets.toml`:

```toml
GEMINI_API_KEY = "your-key"
```

Without a key the app runs in offline mode. Answers still come from the
simulation core with real computed values, but there is no natural language
understanding.

## Roadmap

- Live sensor ingestion over Modbus TCP / BACnet from the rig PLC
- WebXR headset build of the AHU environment
- Student quiz module with export to institutional LMS
- Historical run logging for measured vs theoretical comparison
