"""
AI-Enabled Intelligent HVAC Learning Platform
=============================================
One simulation core (the AHU digital twin) feeding three work packages:

  WP1 - Immersive Virtual Learning Environment  -> interactive AHU schematic
  WP2 - AI Learning Assistant                   -> context-aware LLM tutor
  WP3 - Intelligent Psychrometric Visualization -> ASHRAE psychrometric chart

All psychrometric properties are computed with PsychroLib, an implementation of
the ASHRAE Handbook of Fundamentals (2017) formulations. No hand-rolled
correlations are used anywhere in this file.
"""

import os
import json
import streamlit as st
import streamlit.components.v1 as components
import plotly.graph_objects as go
import psychrolib as psy

psy.SetUnitSystem(psy.SI)

P_ATM = 101325.0          # Pa, sea level
CP_AIR = 1.006            # kJ/kg.K, dry air
CP_VAP = 1.860            # kJ/kg.K, water vapour
T_HIGH_LIMIT = 60.0       # deg C, electric heater cutout

st.set_page_config(page_title="AI HVAC Learning Platform", layout="wide")


# ==========================================================================
# SIMULATION CORE - the AHU digital twin
# ==========================================================================

def state(t_db, w, label, tag=""):
    """Build a full psychrometric state from dry bulb temperature and
    humidity ratio. Every property below comes from PsychroLib / ASHRAE."""
    t_db = min(max(t_db, -50.0), 120.0)     # keep inside PsychroLib's domain
    w = max(w, 1e-6)
    # Above ~100 C at 1 atm the saturation pressure exceeds atmospheric and the
    # saturation humidity ratio goes negative, so the clamp is only meaningful
    # below boiling.
    if t_db < 99.0:
        w_sat = psy.GetSatHumRatio(t_db, P_ATM)
        if w_sat > 0:
            w = min(w, w_sat)               # cannot exceed saturation
    rh = psy.GetRelHumFromHumRatio(t_db, w, P_ATM)
    return {
        "label": label,
        "tag": tag,
        "t_db": t_db,
        "w": w,
        "rh": rh * 100.0,
        "h": psy.GetMoistAirEnthalpy(t_db, w) / 1000.0,      # kJ/kg
        "t_dp": psy.GetTDewPointFromHumRatio(t_db, w, P_ATM),
        "t_wb": psy.GetTWetBulbFromHumRatio(t_db, w, P_ATM),
        "v": psy.GetMoistAirVolume(t_db, w, P_ATM),          # m3/kg
    }


def simulate(t_intake, rh_intake, airflow, t_chw, reheat_kw, humid_kgh, faults):
    """Run air through the AHU and return the state at every sensor location.

    Process chain (mirrors the lab rig on slide 3):
        ST-1/SH-1  intake
        ST-5/SH-3  after cooling coil   (cool + dehumidify)
        ST-7/SH-4  after reheat         (sensible only)
        ST-9/SH-5  after humidifier     (latent only)
    """

    # ---- fault effects on the physical plant -----------------------------
    bypass = 0.12                 # clean coil bypass factor
    approach = 1.5                # K, chilled water to coil surface
    flow_factor = 1.0

    if "Fouled cooling coil" in faults:
        bypass = 0.38             # less contact area -> more air bypasses
        approach = 4.5
    if "Clogged air filter" in faults:
        flow_factor *= 0.55
    if "Fan belt slipping" in faults:
        flow_factor *= 0.70
    if "Low chilled water flow" in faults:
        approach += 5.0

    airflow_actual = airflow * flow_factor

    # ---- 1. intake -------------------------------------------------------
    w_in = psy.GetHumRatioFromRelHum(t_intake, rh_intake / 100.0, P_ATM)
    s1 = state(t_intake, w_in, "Intake air", "ST-1 / SH-1")

    # mass flow from actual (not standard) specific volume
    m_dot = (airflow_actual / 3600.0) / s1["v"]        # kg/s dry air basis

    # ---- 2. cooling coil : apparatus dew point + bypass factor model -----
    t_adp = t_chw + approach
    w_adp = psy.GetSatHumRatio(t_adp, P_ATM)

    t2 = t_adp + bypass * (s1["t_db"] - t_adp)
    if w_adp < s1["w"]:
        # coil surface is below intake dew point -> condensation occurs
        w2 = w_adp + bypass * (s1["w"] - w_adp)
        dehumidifying = True
    else:
        # dry coil, sensible cooling only
        w2 = s1["w"]
        dehumidifying = False
    t2 = min(t2, s1["t_db"])
    s2 = state(t2, w2, "After cooling coil", "ST-5 / SH-3")

    # ---- 3. reheat : sensible heating, humidity ratio unchanged ----------
    cp_moist = CP_AIR + CP_VAP * s2["w"]
    dt_reheat = reheat_kw / (m_dot * cp_moist) if m_dot > 1e-6 else 0.0
    # Electric duct heaters carry a high-limit thermostat. If airflow is too low
    # for the selected duty the element would glow and the cutout opens, so we
    # cap the leaving air temperature rather than letting dT run away.
    t_reheat = s2["t_db"] + dt_reheat
    heater_tripped = t_reheat > T_HIGH_LIMIT
    if heater_tripped:
        t_reheat = T_HIGH_LIMIT
    s3 = state(t_reheat, s2["w"], "After reheat", "ST-7 / SH-4")

    # ---- 4. humidifier : moisture added, dry bulb ~unchanged -------------
    dw = (humid_kgh / 3600.0) / m_dot if m_dot > 1e-6 else 0.0
    s4 = state(s3["t_db"], s3["w"] + dw, "Supply to chamber", "ST-9 / SH-5")

    # ---- coil load breakdown --------------------------------------------
    q_total = m_dot * (s1["h"] - s2["h"])
    q_sens = m_dot * cp_moist * (s1["t_db"] - s2["t_db"])
    q_lat = q_total - q_sens
    shr = q_sens / q_total if abs(q_total) > 1e-6 else 0.0
    condensate = m_dot * (s1["w"] - s2["w"]) * 3600.0      # kg/h

    return {
        "states": [s1, s2, s3, s4],
        "m_dot": m_dot,
        "airflow_actual": airflow_actual,
        "t_adp": t_adp,
        "bypass": bypass,
        "q_total": q_total,
        "q_sens": q_sens,
        "q_lat": q_lat,
        "shr": shr,
        "condensate": max(condensate, 0.0),
        "dehumidifying": dehumidifying,
        "heater_tripped": heater_tripped,
        "dt_reheat_demand": dt_reheat,
        "faults": faults,
    }


def diagnose(sim, airflow_setpoint):
    """Compare the twin against expected behaviour and raise findings.
    This is what makes WP3's 'detect abnormal conditions' concrete."""
    findings = []
    s1, s2 = sim["states"][0], sim["states"][1]

    if sim["bypass"] > 0.25:
        findings.append((
            "error", "Cooling coil underperforming",
            f"Bypass factor is {sim['bypass']:.2f} against a clean-coil baseline "
            f"of 0.12. Air is leaving the coil at {s2['t_db']:.1f} deg C when "
            f"{s1['t_db'] - (s1['t_db'] - sim['t_adp']) * 0.88:.1f} deg C is expected. "
            "Consistent with a fouled or partially blocked coil face."
        ))

    flow_loss = 1.0 - sim["airflow_actual"] / max(airflow_setpoint, 1e-6)
    if flow_loss > 0.15:
        findings.append((
            "error", "Airflow deficit",
            f"Delivered flow is {sim['airflow_actual']:.0f} m3/h against a setpoint of "
            f"{airflow_setpoint:.0f} m3/h, a {flow_loss*100:.0f}% shortfall. "
            "Check filter differential pressure and fan belt tension."
        ))

    if sim["shr"] < 0.55 and sim["q_total"] > 0.1:
        findings.append((
            "warn", "Latent-dominated coil load",
            f"Sensible heat ratio is {sim['shr']:.2f}. The coil is spending most of "
            "its capacity on dehumidification rather than cooling."
        ))

    if s2["rh"] > 95:
        findings.append((
            "warn", "Saturated coil discharge",
            f"Air leaves the coil at {s2['rh']:.0f}% RH. Carryover of water droplets "
            "into the duct is likely; check the eliminator and drain pan."
        ))

    if sim["t_adp"] > s1["t_dp"]:
        findings.append((
            "warn", "No dehumidification occurring",
            f"Apparatus dew point ({sim['t_adp']:.1f} deg C) sits above the intake dew "
            f"point ({s1['t_dp']:.1f} deg C), so the coil is running dry. Lower the "
            "chilled water temperature to remove moisture."
        ))

    if sim["heater_tripped"]:
        findings.append((
            "error", "Reheat high-limit cutout",
            f"The selected reheat duty would raise the air by "
            f"{sim['dt_reheat_demand']:.0f} K at the current mass flow of "
            f"{sim['m_dot']:.3f} kg/s, taking it past the {T_HIGH_LIMIT:.0f} deg C "
            "element cutout. On the real rig the high-limit thermostat opens and "
            "the heater de-energises. Raise airflow or lower the reheat duty."))

    if not findings:
        findings.append((
            "ok", "All monitored parameters nominal",
            f"Coil bypass {sim['bypass']:.2f}, SHR {sim['shr']:.2f}, "
            f"condensate {sim['condensate']:.2f} kg/h. No deviation from the "
            "expected psychrometric process."
        ))
    return findings


# ==========================================================================
# WP3 - PSYCHROMETRIC CHART
# ==========================================================================

def psych_chart(sim, show_process=True):
    fig = go.Figure()
    t_range = [t * 0.5 for t in range(int(0 / 0.5), int(50 / 0.5) + 1)]

    # constant-RH curves, saturation last so it draws on top
    for rh in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
        ws = [psy.GetHumRatioFromRelHum(t, rh / 100.0, P_ATM) * 1000 for t in t_range]
        sat = rh == 100
        fig.add_trace(go.Scatter(
            x=t_range, y=ws, mode="lines", showlegend=False, hoverinfo="skip",
            line=dict(color="#4a90d9" if sat else "#7f8c9a",
                      width=2.5 if sat else 0.8,
                      dash="solid" if sat else "dot"),
            name=f"{rh}% RH",
        ))
        if rh in (20, 40, 60, 80, 100):
            idx = min(range(len(t_range)), key=lambda i: abs(ws[i] - 27))
            if ws[idx] < 27 and t_range[idx] < 49:
                fig.add_annotation(x=t_range[idx], y=ws[idx], text=f"{rh}%",
                                   showarrow=False, font=dict(size=9, color="#7f8c9a"),
                                   xshift=14, yshift=6)

    # constant-enthalpy lines
    for h_val in range(20, 121, 20):
        xs, ys = [], []
        for t in t_range:
            w = (h_val - CP_AIR * t) / (2501 + CP_VAP * t)
            if 0 < w < 0.030 and w <= psy.GetSatHumRatio(t, P_ATM):
                xs.append(t)
                ys.append(w * 1000)
        if xs:
            fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", showlegend=False,
                                     hoverinfo="skip",
                                     line=dict(color="#6b5b8a", width=0.6, dash="dash")))

    # process line through the four state points
    if show_process:
        pts = sim["states"]
        fig.add_trace(go.Scatter(
            x=[p["t_db"] for p in pts], y=[p["w"] * 1000 for p in pts],
            mode="lines", line=dict(color="#ff4b4b", width=3),
            name="Process path", hoverinfo="skip",
        ))
        colors = ["#ffd166", "#4cc9f0", "#f77f00", "#06d6a0"]
        for i, p in enumerate(pts):
            fig.add_trace(go.Scatter(
                x=[p["t_db"]], y=[p["w"] * 1000], mode="markers+text",
                marker=dict(size=15, color=colors[i],
                            line=dict(color="white", width=1.5)),
                text=[str(i + 1)], textposition="middle center",
                textfont=dict(size=10, color="#111"),
                name=f"{i+1}. {p['label']}",
                hovertemplate=(
                    f"<b>{p['label']}</b> ({p['tag']})<br>"
                    f"Dry bulb: {p['t_db']:.1f} deg C<br>"
                    f"Humidity ratio: {p['w']*1000:.2f} g/kg<br>"
                    f"RH: {p['rh']:.1f} %<br>"
                    f"Enthalpy: {p['h']:.2f} kJ/kg<br>"
                    f"Dew point: {p['t_dp']:.1f} deg C<br>"
                    f"Wet bulb: {p['t_wb']:.1f} deg C<extra></extra>"),
            ))
        # apparatus dew point marker
        fig.add_trace(go.Scatter(
            x=[sim["t_adp"]], y=[psy.GetSatHumRatio(sim["t_adp"], P_ATM) * 1000],
            mode="markers", marker=dict(size=11, color="#ff4b4b", symbol="x-thin",
                                        line=dict(width=2.5, color="#ff4b4b")),
            name="Apparatus dew point", hoverinfo="skip",
        ))

    fig.update_layout(
        xaxis=dict(title="Dry Bulb Temperature (deg C)", range=[0, 50],
                   gridcolor="rgba(128,128,128,0.15)", dtick=5),
        yaxis=dict(title="Humidity Ratio (g water / kg dry air)", range=[0, 27],
                   side="right", gridcolor="rgba(128,128,128,0.15)", dtick=5),
        height=430, margin=dict(l=10, r=60, t=30, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0, font=dict(size=10)),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


# ==========================================================================
# WP1 - IMMERSIVE SCHEMATIC
# ==========================================================================

def ahu_svg(sim):
    s1, s2, s3, s4 = sim["states"]
    drip = ""
    if sim["condensate"] > 0.01:
        for i, dx in enumerate([0, 9, 18]):
            drip += (
                f'<circle cx="{243+dx}" cy="150" r="2.6" fill="#4cc9f0">'
                f'<animate attributeName="cy" values="150;196" dur="1.5s" '
                f'begin="{i*0.5}s" repeatCount="indefinite"/>'
                f'<animate attributeName="opacity" values="1;1;0" dur="1.5s" '
                f'begin="{i*0.5}s" repeatCount="indefinite"/></circle>')

    speed = max(0.6, 3.0 * (500.0 / max(sim["airflow_actual"], 60)))

    def badge(x, s, color):
        return f'''
        <rect x="{x}" y="34" width="104" height="46" rx="5"
              fill="rgba(20,24,32,0.92)" stroke="{color}" stroke-width="1.4"/>
        <text x="{x+52}" y="49" font-size="9" fill="{color}"
              text-anchor="middle" font-family="monospace">{s["tag"]}</text>
        <text x="{x+52}" y="63" font-size="12" fill="#fff" text-anchor="middle"
              font-family="monospace" font-weight="bold">{s["t_db"]:.1f} C</text>
        <text x="{x+52}" y="75" font-size="10" fill="#9fb0c0" text-anchor="middle"
              font-family="monospace">{s["rh"]:.0f}% RH</text>'''

    coil_color = "#e63946" if sim["bypass"] > 0.25 else "#4cc9f0"

    return f'''<!DOCTYPE html><html><head><meta charset="utf-8">
    <style>
      html,body {{ margin:0; padding:0; background:transparent; overflow:hidden; }}
      svg {{ display:block; width:100%; height:auto; }}
      text {{ font-family: "Source Sans Pro", system-ui, sans-serif; }}
    </style></head><body>
    <svg viewBox="0 0 860 260" preserveAspectRatio="xMidYMid meet"
         xmlns="http://www.w3.org/2000/svg">
      <defs>
        <marker id="ar" markerWidth="7" markerHeight="7" refX="6" refY="2.4"
                orient="auto"><path d="M0,0 L0,4.8 L6,2.4 z" fill="#6ee7ff"/></marker>
      </defs>

      <!-- duct -->
      <rect x="40" y="95" width="700" height="70" rx="4" fill="rgba(255,255,255,0.03)"
            stroke="#4a5568" stroke-width="1.6"/>

      <!-- airflow streamlines -->
      <path d="M55,115 H735" stroke="#6ee7ff" stroke-width="1.6" fill="none"
            stroke-dasharray="14 10" marker-end="url(#ar)" opacity="0.75">
        <animate attributeName="stroke-dashoffset" values="48;0"
                 dur="{speed:.2f}s" repeatCount="indefinite"/></path>
      <path d="M55,145 H735" stroke="#6ee7ff" stroke-width="1.6" fill="none"
            stroke-dasharray="14 10" marker-end="url(#ar)" opacity="0.75">
        <animate attributeName="stroke-dashoffset" values="48;0"
                 dur="{speed*1.25:.2f}s" repeatCount="indefinite"/></path>

      <!-- blower -->
      <circle cx="112" cy="130" r="26" fill="rgba(110,231,255,0.08)"
              stroke="#6ee7ff" stroke-width="1.6"/>
      <g>
        <path d="M112,112 L118,130 L112,148 L106,130 Z" fill="#6ee7ff" opacity="0.9"/>
        <path d="M94,130 L112,124 L130,130 L112,136 Z" fill="#6ee7ff" opacity="0.6"/>
        <circle cx="112" cy="130" r="4" fill="#6ee7ff"/>
        <animateTransform attributeName="transform" type="rotate"
                          from="0 112 130" to="360 112 130"
                          dur="{speed*0.35:.2f}s" repeatCount="indefinite"
                          additive="sum"/>
      </g>
      <text x="112" y="184" font-size="10" fill="#9fb0c0" text-anchor="middle">BLOWER AVE-1</text>
      <text x="112" y="196" font-size="9" fill="#6ee7ff" text-anchor="middle"
            font-family="monospace">{sim["airflow_actual"]:.0f} m3/h</text>

      <!-- cooling coil -->
      <rect x="228" y="98" width="62" height="64" rx="3" fill="rgba(76,201,240,0.14)"
            stroke="{coil_color}" stroke-width="2"/>
      {''.join(f'<line x1="{234+i*9}" y1="102" x2="{234+i*9}" y2="158" stroke="{coil_color}" stroke-width="1.6" opacity="0.65"/>' for i in range(7))}
      <text x="259" y="184" font-size="10" fill="#9fb0c0" text-anchor="middle">COOLING COIL</text>
      <text x="259" y="196" font-size="9" fill="{coil_color}" text-anchor="middle"
            font-family="monospace">ADP {sim["t_adp"]:.1f} C</text>
      {drip}
      <text x="259" y="216" font-size="9" fill="#4cc9f0" text-anchor="middle"
            font-family="monospace">{sim["condensate"]:.2f} kg/h condensate</text>

      <!-- reheat -->
      <rect x="410" y="98" width="62" height="64" rx="3" fill="rgba(230,57,70,0.14)"
            stroke="#e63946" stroke-width="2"/>
      {''.join(f'<line x1="416" y1="{106+i*13}" x2="466" y2="{106+i*13}" stroke="#e63946" stroke-width="2" opacity="0.75"/>' for i in range(5))}
      <text x="441" y="184" font-size="10" fill="#9fb0c0" text-anchor="middle">HEATER AR-1</text>

      <!-- humidifier -->
      <rect x="580" y="98" width="62" height="64" rx="3" fill="rgba(6,214,160,0.14)"
            stroke="#06d6a0" stroke-width="2"/>
      {''.join(f'<circle cx="{594+i*16}" cy="{118+(i%2)*22}" r="4" fill="#06d6a0" opacity="0.7"><animate attributeName="r" values="2;6;2" dur="2s" begin="{i*0.4}s" repeatCount="indefinite"/></circle>' for i in range(3))}
      <text x="611" y="184" font-size="10" fill="#9fb0c0" text-anchor="middle">HUMIDIFIER AHUM-1</text>

      <!-- chamber -->
      <rect x="748" y="72" width="76" height="116" rx="4" fill="rgba(255,255,255,0.04)"
            stroke="#4a5568" stroke-width="1.6"/>
      <text x="786" y="134" font-size="10" fill="#9fb0c0" text-anchor="middle">TEST</text>
      <text x="786" y="147" font-size="10" fill="#9fb0c0" text-anchor="middle">CHAMBER</text>

      {badge(46, s1, "#ffd166")}
      {badge(300, s2, "#4cc9f0")}
      {badge(482, s3, "#f77f00")}
      {badge(652, s4, "#06d6a0")}
    </svg></body></html>'''


# ==========================================================================
# WP2 - AI LEARNING ASSISTANT
# ==========================================================================

SYSTEM_PROMPT = """You are the AI Lab Assistant for a Singapore Institute of \
Technology HVAC teaching rig (an air handling unit with a cooling coil, electric \
reheater, steam humidifier and test chamber).

You are talking to an undergraduate engineering student standing at the rig.

BOUNDARY CONDITIONS - these are strict:
- Only discuss HVAC, thermodynamics, psychrometrics, and this laboratory.
  If asked about anything else, politely decline and steer back to the lab.
- The live state below is your ONLY source of measured data. Never invent a
  sensor reading. If a value is not in the data, say you do not have that sensor.
- All psychrometric values were computed with PsychroLib to ASHRAE
  Fundamentals formulations. Trust them over your own mental arithmetic.
- Teach rather than just answer. Where useful, end with a short question that
  pushes the student to predict what happens next.
- Keep replies under 150 words. Answer directly and completely in one pass;
  do not deliberate at length. Use plain text, no LaTeX.

LIVE SYSTEM STATE (JSON):
{context}
"""


def build_context(sim, controls):
    return json.dumps({
        "controls": controls,
        "injected_faults": sim["faults"] or ["none"],
        "mass_flow_kg_s": round(sim["m_dot"], 4),
        "apparatus_dew_point_C": round(sim["t_adp"], 2),
        "coil_bypass_factor": round(sim["bypass"], 3),
        "coil_load_kW": {
            "total": round(sim["q_total"], 3),
            "sensible": round(sim["q_sens"], 3),
            "latent": round(sim["q_lat"], 3),
            "sensible_heat_ratio": round(sim["shr"], 3),
        },
        "condensate_kg_per_h": round(sim["condensate"], 3),
        "state_points": [{
            "point": i + 1, "label": s["label"], "sensor_tag": s["tag"],
            "dry_bulb_C": round(s["t_db"], 2),
            "humidity_ratio_g_kg": round(s["w"] * 1000, 3),
            "relative_humidity_pct": round(s["rh"], 1),
            "enthalpy_kJ_kg": round(s["h"], 2),
            "dew_point_C": round(s["t_dp"], 2),
            "wet_bulb_C": round(s["t_wb"], 2),
            "specific_volume_m3_kg": round(s["v"], 4),
        } for i, s in enumerate(sim["states"])],
    }, indent=1)


def get_api_key():
    try:
        if "GEMINI_API_KEY" in st.secrets:
            return st.secrets["GEMINI_API_KEY"]
    except Exception:
        pass
    return os.environ.get("GEMINI_API_KEY", "")


def extract_text(resp):
    """Pull the visible answer out of a response. resp.text can be None when the
    model spent its budget on thinking, so fall back to walking the parts."""
    try:
        if getattr(resp, "text", None):
            return resp.text.strip()
    except Exception:
        pass
    chunks = []
    for cand in getattr(resp, "candidates", None) or []:
        content = getattr(cand, "content", None)
        for part in (getattr(content, "parts", None) or []):
            if getattr(part, "thought", False):
                continue
            if getattr(part, "text", None):
                chunks.append(part.text)
    return "\n".join(chunks).strip()


def resolve_model():
    """Ask the API which models this key can actually use.

    Model IDs churn (Gemini 2.0 Flash was retired on 1 June 2026), so hard-coding
    one is fragile. We list what the key can reach, keep the text models that
    support generateContent, and prefer Flash tiers by generation.
    """
    if "model_id" in st.session_state:
        return st.session_state.model_id

    key = get_api_key()
    if not key:
        return None
    try:
        from google import genai
        client = genai.Client(api_key=key)
        usable = [
            m.name.replace("models/", "")
            for m in client.models.list()
            if "generateContent" in getattr(m, "supported_actions", []) or
               "generateContent" in getattr(m, "supported_generation_methods", [])
        ]
        # exclude non-text variants that would fail on a plain text prompt
        usable = [m for m in usable if not any(
            bad in m for bad in ("image", "video", "audio", "tts", "embedding",
                                 "veo", "imagen", "live", "native-audio"))]

        def rank(name):
            flash = "flash" in name
            lite = "lite" in name
            preview = "preview" in name or "exp" in name
            # highest version number first, prefer stable non-lite flash
            digits = "".join(c if c.isdigit() else " " for c in name).split()
            ver = float(digits[0]) if digits else 0
            return (flash, not preview, not lite, ver)

        usable.sort(key=rank, reverse=True)
        if usable:
            st.session_state.model_id = usable[0]
            st.session_state.model_options = usable[:12]
            return usable[0]
    except Exception as e:
        st.session_state.model_error = str(e)
    return None


def ask_llm(question, sim, controls, history):
    """Live Gemini call with the digital twin state injected as context."""
    key = get_api_key()
    if not key:
        return None, "no_key"

    model_id = resolve_model()
    if not model_id:
        return None, "no_model"

    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=key)
        contents = []
        for m in history[-8:]:
            contents.append(types.Content(
                role="user" if m["role"] == "user" else "model",
                parts=[types.Part(text=m["content"])]))
        contents.append(types.Content(role="user", parts=[types.Part(text=question)]))

        sys_prompt = SYSTEM_PROMPT.format(context=build_context(sim, controls))

        def make_config(disable_thinking):
            """Gemini 3.x models reason before answering and those thinking
            tokens are charged against max_output_tokens. Left unchecked they
            consume the whole budget and the visible answer is truncated
            mid-sentence. We ask for minimal thinking and leave ample headroom."""
            kw = dict(system_instruction=sys_prompt, temperature=0.4,
                      max_output_tokens=2048)
            if disable_thinking:
                kw["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
            return types.GenerateContentConfig(**kw)

        try:
            resp = client.models.generate_content(
                model=model_id, contents=contents, config=make_config(True))
        except Exception:
            # some models require thinking and reject a zero budget
            resp = client.models.generate_content(
                model=model_id, contents=contents, config=make_config(False))

        text = extract_text(resp)
        if not text:
            return None, "empty"
        return text, "ok"
    except Exception as e:
        msg = str(e)
        # a retired or unavailable model returns 404 / "limit: 0" - try the next one
        if ("404" in msg or "limit: 0" in msg) and "model_options" in st.session_state:
            opts = st.session_state.model_options
            if model_id in opts and opts.index(model_id) + 1 < len(opts):
                st.session_state.model_id = opts[opts.index(model_id) + 1]
                return ask_llm(question, sim, controls, history)
        st.session_state.last_api_error = msg
        return None, "unavailable"


def offline_answer(question, sim, controls):
    """Deterministic fallback. Unlike a keyword matcher, every number
    here is read from the simulation core rather than invented."""
    q = question.lower()
    s1, s2, s3, s4 = sim["states"]

    if any(k in q for k in ["enthalpy", "total heat", "energy"]):
        return (f"Enthalpy at each point: intake {s1['h']:.2f}, after coil "
                f"{s2['h']:.2f}, after reheat {s3['h']:.2f}, supply {s4['h']:.2f} kJ/kg. "
                f"The coil removes {sim['q_total']:.2f} kW total, split "
                f"{sim['q_sens']:.2f} kW sensible and {sim['q_lat']:.2f} kW latent "
                f"(SHR {sim['shr']:.2f}). Why does reheat raise enthalpy but not "
                "humidity ratio?")
    if any(k in q for k in ["dew", "condens", "moisture", "water"]):
        if sim["dehumidifying"]:
            return (f"The intake dew point is {s1['t_dp']:.1f} deg C and the apparatus "
                    f"dew point is {sim['t_adp']:.1f} deg C. Since the coil surface is "
                    f"colder than the dew point, moisture condenses at "
                    f"{sim['condensate']:.2f} kg/h. Humidity ratio falls from "
                    f"{s1['w']*1000:.2f} to {s2['w']*1000:.2f} g/kg.")
        return (f"No condensation right now. The apparatus dew point "
                f"({sim['t_adp']:.1f} deg C) is above the intake dew point "
                f"({s1['t_dp']:.1f} deg C), so the coil runs dry and cools "
                "sensibly only. What would you change to start dehumidifying?")
    if any(k in q for k in ["fault", "diagnos", "wrong", "problem", "alarm"]):
        return " ".join(f"[{lvl.upper()}] {t}: {d}" for lvl, t, d in
                        diagnose(sim, controls["airflow_setpoint"]))
    if any(k in q for k in ["shr", "sensible", "latent"]):
        return (f"Sensible load {sim['q_sens']:.2f} kW, latent {sim['q_lat']:.2f} kW, "
                f"total {sim['q_total']:.2f} kW, giving SHR {sim['shr']:.2f}. "
                "Sensible heat changes dry bulb temperature; latent heat changes "
                "moisture content at constant temperature.")
    if any(k in q for k in ["coil", "chill", "adp", "bypass"]):
        return (f"The coil is modelled with an apparatus dew point of "
                f"{sim['t_adp']:.1f} deg C and a bypass factor of {sim['bypass']:.2f}. "
                f"Air leaves at {s2['t_db']:.1f} deg C, {s2['rh']:.0f}% RH. Bypass "
                "factor is the fraction of air that passes through untreated.")
    if any(k in q for k in ["airflow", "flow", "blower", "fan"]):
        return (f"Delivered airflow is {sim['airflow_actual']:.0f} m3/h against a "
                f"setpoint of {controls['airflow_setpoint']:.0f} m3/h, giving a mass "
                f"flow of {sim['m_dot']:.3f} kg/s using the intake specific volume of "
                f"{s1['v']:.4f} m3/kg.")
    tail = ("" if get_api_key() else
            " [Offline mode - add a GEMINI_API_KEY to enable the full tutor.]")
    return (f"Current supply air is {s4['t_db']:.1f} deg C at {s4['rh']:.0f}% RH "
            f"({s4['w']*1000:.2f} g/kg, {s4['h']:.2f} kJ/kg). Ask me about the coil "
            f"load, dew point, sensible heat ratio, or run a diagnostic.{tail}")


# ==========================================================================
# UI
# ==========================================================================

st.title("AI-Enabled Intelligent HVAC Learning Platform")
st.caption("Real-Time Digital Twin | ASHRAE Psychrometrics | AI Assistant & Diagnostics")

with st.sidebar:
    st.header("Live Sensor Controls")
    st.caption("Intake conditions")
    t_intake = st.slider("Intake Dry Bulb (deg C)", 15.0, 45.0, 32.0, 0.5)
    rh_intake = st.slider("Intake Relative Humidity (%)", 20.0, 95.0, 70.0, 1.0)
    airflow = st.slider("Airflow Setpoint (m3/h)", 100, 1000, 500, 25)

    st.caption("Plant setpoints")
    t_chw = st.slider("Chilled Water Temp (deg C)", 4.0, 20.0, 7.0, 0.5)
    reheat_kw = st.slider("Reheat Duty (kW)", 0.0, 5.0, 1.5, 0.1)
    humid_kgh = st.slider("Humidifier Output (kg/h)", 0.0, 6.0, 0.5, 0.1)

    st.divider()
    st.subheader("Fault Injection")
    st.caption("Instructor mode: degrade the plant and see if students catch it.")
    faults = st.multiselect(
        "Inject fault", ["Fouled cooling coil", "Clogged air filter",
                         "Fan belt slipping", "Low chilled water flow"],
        label_visibility="collapsed")

    st.divider()
    if st.session_state.get("last_api_error"):
        with st.expander("Tutor connection details"):
            st.caption(st.session_state["last_api_error"][:600])
    st.caption("Psychrometrics: PsychroLib (ASHRAE Fundamentals 2017)")
    if get_api_key():
        _m = resolve_model()
        st.caption(f"AI tutor: Gemini connected ({_m})" if _m
                   else "AI tutor: key present but no usable model found")
    else:
        st.caption("AI tutor: OFFLINE - no key")

controls = {
    "intake_dry_bulb_C": t_intake, "intake_RH_pct": rh_intake,
    "airflow_setpoint": airflow, "chilled_water_C": t_chw,
    "reheat_kW": reheat_kw, "humidifier_kg_h": humid_kgh,
}
sim = simulate(t_intake, rh_intake, airflow, t_chw, reheat_kw, humid_kgh, faults)
findings = diagnose(sim, airflow)

# ---- diagnostics banner --------------------------------------------------
st.subheader("Predictive Diagnostics & Anomaly Detection")
for lvl, title, detail in findings:
    {"error": st.error, "warn": st.warning, "ok": st.success}[lvl](
        f"**{title}** - {detail}")

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Supply Air", f"{sim['states'][3]['t_db']:.1f} C",
          f"{sim['states'][3]['t_db'] - t_intake:+.1f} C vs intake")
k2.metric("Supply RH", f"{sim['states'][3]['rh']:.0f} %")
k3.metric("Coil Load", f"{sim['q_total']:.2f} kW",
          f"SHR {sim['shr']:.2f}", delta_color="off")
k4.metric("Condensate", f"{sim['condensate']:.2f} kg/h")
k5.metric("Mass Flow", f"{sim['m_dot']:.3f} kg/s")

st.divider()

tab3, tab1, tab2 = st.tabs([
    "WP3 - Psychrometric Visualization",
    "WP1 - Immersive AHU Environment",
    "WP2 - AI Learning Assistant"])

# ---- WP3 ------------------------------------------------------------------
with tab3:
    c1, c2 = st.columns([3, 2])
    with c1:
        st.plotly_chart(psych_chart(sim), use_container_width=True)
        st.caption("Solid blue = saturation curve. Dotted grey = constant RH. "
                   "Dashed purple = constant enthalpy. Red X = apparatus dew point.")
    with c2:
        st.markdown("**State point properties**")
        st.dataframe([{
            "Pt": i + 1, "Location": s["label"], "Tag": s["tag"],
            "T db (C)": round(s["t_db"], 1), "RH (%)": round(s["rh"], 1),
            "W (g/kg)": round(s["w"] * 1000, 2), "h (kJ/kg)": round(s["h"], 2),
            "T dp (C)": round(s["t_dp"], 1), "T wb (C)": round(s["t_wb"], 1),
            "v (m3/kg)": round(s["v"], 4),
        } for i, s in enumerate(sim["states"])], hide_index=True,
            use_container_width=True)

        st.markdown("**Cooling coil heat transfer**")
        st.dataframe([
            {"Quantity": "Total (latent + sensible)", "Value": f"{sim['q_total']:.3f} kW"},
            {"Quantity": "Sensible", "Value": f"{sim['q_sens']:.3f} kW"},
            {"Quantity": "Latent", "Value": f"{sim['q_lat']:.3f} kW"},
            {"Quantity": "Sensible heat ratio", "Value": f"{sim['shr']:.3f}"},
            {"Quantity": "Apparatus dew point", "Value": f"{sim['t_adp']:.2f} C"},
            {"Quantity": "Bypass factor", "Value": f"{sim['bypass']:.3f}"},
        ], hide_index=True, use_container_width=True)

        st.download_button(
            "Export analysis (JSON)", build_context(sim, controls),
            file_name="hvac_analysis.json", mime="application/json",
            use_container_width=True)

# ---- WP1 ------------------------------------------------------------------
with tab1:
    st.markdown("**Live AHU cutaway** - airflow speed, coil condensate and every "
                "sensor badge are driven by the same simulation core as the chart.")
    # components.v1.html renders in an iframe with no sanitisation, so the
    # SVG animation tags survive. st.html() strips them and shows nothing.
    components.html(ahu_svg(sim), height=290, scrolling=False)
    st.caption("Streamline animation rate scales with delivered airflow. Droplets "
               "appear only when the coil surface falls below the intake dew point. "
               "The coil outline turns red when its bypass factor degrades.")

    with st.expander("Component walkthrough"):
        for s, txt in zip(sim["states"], [
            "Ambient air is drawn in past ST-1/SH-1. This is state point 1 and fixes "
            "the intake dew point, which decides whether the coil can dehumidify.",
            "The cooling coil cools air toward its apparatus dew point. Air that "
            "contacts the fins is saturated at ADP; the rest bypasses untreated. "
            "Mixing the two gives the outlet state.",
            "The electric reheater adds sensible heat only. On the chart this moves "
            "horizontally to the right: humidity ratio is unchanged, RH falls.",
            "The humidifier injects steam, raising humidity ratio at nearly constant "
            "dry bulb. On the chart this is a near-vertical rise."]):
            st.markdown(f"**{s['label']}** ({s['tag']}) - {s['t_db']:.1f} deg C, "
                        f"{s['rh']:.0f}% RH, {s['w']*1000:.2f} g/kg  \n{txt}")

# ---- WP2 ------------------------------------------------------------------
with tab2:
    hcol, bcol = st.columns([4, 1])
    hcol.markdown("**Context-aware tutor.** The full digital twin state is injected "
                  "into the model's context on every turn, so answers cite live "
                  "measured values rather than generic HVAC theory.")
    if bcol.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    if not get_api_key():
        st.info("Running in offline mode. Add GEMINI_API_KEY to Streamlit secrets "
                "or the environment to enable the live LLM tutor. Answers below "
                "still use real computed values from the simulation core.")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    box = st.container(height=340)
    with box:
        if not st.session_state.messages:
            st.chat_message("assistant").write(
                "I can see the live rig state. Ask me why the coil is or is not "
                "dehumidifying, what the sensible heat ratio means, or tell me to "
                "run a diagnostic.")
        for m in st.session_state.messages:
            st.chat_message(m["role"]).write(m["content"])

    if prompt := st.chat_input("Ask about the coil, dew point, SHR, or run a diagnostic..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with box:
            st.chat_message("user").write(prompt)
        reply, status = ask_llm(prompt, sim, controls, st.session_state.messages[:-1])
        if reply is None:
            reply = offline_answer(prompt, sim, controls)
            if status == "empty":
                reply += ("\n\n*(Live tutor returned no text - answered "
                          "from the simulation core.)*")
            elif status == "unavailable":
                reply += ("\n\n*(Live tutor temporarily unavailable - answered "
                          "directly from the simulation core.)*")
        st.session_state.messages.append({"role": "assistant", "content": reply})
        with box:
            st.chat_message("assistant").write(reply)
