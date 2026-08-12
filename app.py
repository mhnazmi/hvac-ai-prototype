"""
AI-Enabled Intelligent HVAC Learning Platform
=============================================
One simulation core (the AHU digital twin) feeding three work packages:

  WP1 - Immersive Virtual Learning Environment  -> interactive AHU schematic
  WP2 - AI Learning Assistant                   -> context-aware LLM tutor
  WP3 - Intelligent Psychrometric Visualization -> ASHRAE psychrometric chart

All psychrometric properties are computed with PsychroLib, an implementation of
the ASHRAE Handbook of Fundamentals (2017) formulations. No hand-rolled
correlations are used for any reported state property.
"""

import os
import io
import csv
import time
import json
import copy
import datetime
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
# PSYCHROMETRIC STATE  (verified - unchanged)
# ==========================================================================

def state(t_db, w, label, tag=""):
    """Build a full psychrometric state from dry bulb temperature and
    humidity ratio. Every property below comes from PsychroLib / ASHRAE."""
    t_db = min(max(t_db, -50.0), 120.0)     # keep inside PsychroLib's domain
    w = max(w, 1e-6)
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
    bypass = 0.12                 # clean coil bypass factor
    approach = 1.5                # K, chilled water to coil surface
    flow_factor = 1.0

    if "Fouled cooling coil" in faults:
        bypass = 0.38
        approach = 4.5
    if "Clogged air filter" in faults:
        flow_factor *= 0.55
    if "Fan belt slipping" in faults:
        flow_factor *= 0.70
    if "Low chilled water flow" in faults:
        approach += 5.0

    airflow_actual = airflow * flow_factor

    # 1. intake
    w_in = psy.GetHumRatioFromRelHum(t_intake, rh_intake / 100.0, P_ATM)
    s1 = state(t_intake, w_in, "Intake air", "ST-1 / SH-1")

    m_dot = (airflow_actual / 3600.0) / s1["v"]        # kg/s dry air basis

    # 2. cooling coil : apparatus dew point + bypass factor model
    t_adp = t_chw + approach
    w_adp = psy.GetSatHumRatio(t_adp, P_ATM)

    t2 = t_adp + bypass * (s1["t_db"] - t_adp)
    if w_adp < s1["w"]:
        w2 = w_adp + bypass * (s1["w"] - w_adp)
        dehumidifying = True
    else:
        w2 = s1["w"]
        dehumidifying = False
    t2 = min(t2, s1["t_db"])
    s2 = state(t2, w2, "After cooling coil", "ST-5 / SH-3")

    # 3. reheat : sensible heating, humidity ratio unchanged
    cp_moist = CP_AIR + CP_VAP * s2["w"]
    dt_reheat = reheat_kw / (m_dot * cp_moist) if m_dot > 1e-6 else 0.0
    t_reheat = s2["t_db"] + dt_reheat
    heater_tripped = t_reheat > T_HIGH_LIMIT
    if heater_tripped:
        t_reheat = T_HIGH_LIMIT
    s3 = state(t_reheat, s2["w"], "After reheat", "ST-7 / SH-4")

    # 4. humidifier : moisture added, dry bulb ~unchanged
    dw = (humid_kgh / 3600.0) / m_dot if m_dot > 1e-6 else 0.0
    s4 = state(s3["t_db"], s3["w"] + dw, "Supply to chamber", "ST-9 / SH-5")

    # coil load breakdown
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
        "approach": approach,
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


# ==========================================================================
# PLANT DIAGNOSTICS  (verified - unchanged)
# ==========================================================================

def diagnose(sim, airflow_setpoint):
    """Compare the twin against expected behaviour and raise findings."""
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
# SENSOR-FAULT INJECTION + DETECTION  (WP3: "potential sensor faults")
# ==========================================================================
# Plant faults degrade the physics. Sensor faults corrupt only the *reported*
# reading while the true physics is unchanged, so they are caught by physical
# consistency checks rather than by the value being "high" or "low".

SENSOR_FAULTS = {
    "SH-4 humidity sensor drift": "reheat_w",
    "ST-5 coil-outlet temp bias": "coil_t",
    "SH-3 coil-outlet RH stuck high": "coil_rh",
}


def apply_sensor_faults(sim, sensor_faults):
    """Return a copy of the reported state points with the selected sensor
    faults injected, plus a per-point map of which readings are corrupted."""
    import copy
    reported = copy.deepcopy(sim["states"])
    corrupted = {i: [] for i in range(len(reported))}

    if "SH-4 humidity sensor drift" in sensor_faults:
        # Reheat is sensible-only, so W at pt 3 must equal W at pt 2. A drifting
        # humidity sensor reports a different value - physically impossible.
        reported[2]["w"] = reported[2]["w"] + 0.0022
        reported[2]["rh"] = psy.GetRelHumFromHumRatio(
            reported[2]["t_db"], reported[2]["w"], P_ATM) * 100.0
        corrupted[2].append("w")

    if "ST-5 coil-outlet temp bias" in sensor_faults:
        # Reports the coil outlet colder than the apparatus dew point, which no
        # real coil can achieve (air cannot leave colder than the ADP).
        reported[1]["t_db"] = sim["t_adp"] - 2.5
        corrupted[1].append("t_db")

    if "SH-3 coil-outlet RH stuck high" in sensor_faults:
        reported[1]["rh"] = 101.5     # impossible: RH cannot exceed 100%
        corrupted[1].append("rh")

    return reported, corrupted


def diagnose_sensors(sim, reported):
    """Detect sensor faults from physical inconsistency, not magnitude."""
    findings = []
    s2_true, s3_true = sim["states"][1], sim["states"][2]
    r2, r3 = reported[1], reported[2]

    # RH can never physically exceed saturation
    for i, r in enumerate(reported):
        if r["rh"] > 100.5:
            findings.append((
                "error", f"Sensor fault: {r['tag']}",
                f"Reported RH is {r['rh']:.0f}% at point {i+1}. Relative humidity "
                "cannot exceed 100%; the humidity sensor is reading out of range."))

    # Reheat conserves humidity ratio (sensible only)
    if abs(r3["w"] - r2["w"]) > 0.0005 and abs(s3_true["w"] - s2_true["w"]) < 1e-9:
        findings.append((
            "error", "Sensor fault: SH-4 humidity sensor",
            f"Reported humidity ratio rises {(r3['w']-r2['w'])*1000:.2f} g/kg across "
            "the reheater, but reheat adds sensible heat only and cannot change "
            "moisture content. The SH-4 humidity sensor has drifted."))

    # Air cannot leave the coil colder than the apparatus dew point
    if r2["t_db"] < sim["t_adp"] - 0.3:
        findings.append((
            "error", "Sensor fault: ST-5 temperature sensor",
            f"Reported coil-outlet temperature is {r2['t_db']:.1f} deg C, below the "
            f"apparatus dew point of {sim['t_adp']:.1f} deg C. Air cannot leave the "
            "coil colder than the ADP, so ST-5 is reading with a negative bias."))

    if not findings:
        findings.append((
            "ok", "Sensor integrity nominal",
            "All reported readings are physically consistent with the psychrometric "
            "process. No sensor drift, bias or out-of-range values detected."))
    return findings


# ==========================================================================
# DECISION SUPPORT / OPTIMISATION HINTS  (WP3: operation & optimisation)
# ==========================================================================

def recommend(sim, controls):
    """Rule-based setpoint advice. Each hint names an action and its effect."""
    hints = []
    s1, s2, s3, s4 = sim["states"]

    if "Fouled cooling coil" in sim["faults"]:
        recoverable = sim["q_total"] * (0.38 - 0.12) / 0.38
        hints.append(("high",
            f"Clean the cooling coil. Restoring the bypass factor from "
            f"{sim['bypass']:.2f} to 0.12 recovers roughly {recoverable:.1f} kW of "
            "coil capacity and sharpens the approach temperature."))

    if sim["airflow_actual"] < controls["airflow_setpoint"] * 0.85:
        hints.append(("high",
            f"Restore airflow. Delivered flow ({sim['airflow_actual']:.0f} m3/h) is "
            f"well below the {controls['airflow_setpoint']:.0f} m3/h setpoint; clear "
            "the filter or retension the fan belt before trusting coil performance."))

    if not sim["dehumidifying"] and s1["t_dp"] > 8:
        drop = sim["t_adp"] - s1["t_dp"]
        hints.append(("med",
            f"To begin dehumidification, lower the chilled-water temperature by about "
            f"{drop + 1:.0f} deg C so the apparatus dew point drops below the intake "
            f"dew point of {s1['t_dp']:.1f} deg C."))

    if controls["reheat_kW"] > 0.1 and sim["shr"] < 0.7 and s2["t_db"] < 14:
        hints.append(("med",
            "You are overcooling then reheating. Raising the chilled-water temperature "
            "slightly cuts the coil load and the reheat needed to correct it - the "
            "same supply condition for less total energy."))

    if sim["heater_tripped"]:
        hints.append(("high",
            f"Reheat is on its high-limit cutout. Raise airflow above "
            f"{sim['airflow_actual']:.0f} m3/h or cut the reheat duty so the leaving "
            f"air stays under {T_HIGH_LIMIT:.0f} deg C."))

    if not hints:
        hints.append(("ok",
            "Operating point is efficient for the current targets. Coil load "
            f"{sim['q_total']:.1f} kW at SHR {sim['shr']:.2f}, no wasted reheat, "
            "airflow at setpoint."))
    return hints


# ==========================================================================
# PSYCHROMETRIC CHART  (WP3) - now overlays theoretical vs actual
# ==========================================================================

def psych_chart(sim, sim_ideal=None, selected_pt=None):
    fig = go.Figure()
    t_range = [t * 0.5 for t in range(0, int(50 / 0.5) + 1)]

    for rh in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
        ws = [psy.GetHumRatioFromRelHum(t, rh / 100.0, P_ATM) * 1000 for t in t_range]
        sat = rh == 100
        fig.add_trace(go.Scatter(
            x=t_range, y=ws, mode="lines", showlegend=False, hoverinfo="skip",
            line=dict(color="#4a90d9" if sat else "#7f8c9a",
                      width=2.5 if sat else 0.8,
                      dash="solid" if sat else "dot")))
        if rh in (20, 40, 60, 80, 100):
            idx = min(range(len(t_range)), key=lambda i: abs(ws[i] - 27))
            if ws[idx] < 27 and t_range[idx] < 49:
                fig.add_annotation(x=t_range[idx], y=ws[idx], text=f"{rh}%",
                                   showarrow=False, font=dict(size=9, color="#7f8c9a"),
                                   xshift=14, yshift=6)

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

    # theoretical (clean-plant) process - dashed grey-green, drawn first
    if sim_ideal is not None:
        ip = sim_ideal["states"]
        fig.add_trace(go.Scatter(
            x=[p["t_db"] for p in ip], y=[p["w"] * 1000 for p in ip],
            mode="lines+markers", line=dict(color="#06d6a0", width=2, dash="dash"),
            marker=dict(size=9, color="#06d6a0", symbol="circle-open",
                        line=dict(width=1.5)),
            name="Theoretical (clean plant)",
            hovertemplate="Theoretical<br>%{x:.1f} C, %{y:.2f} g/kg<extra></extra>"))

    # actual (measured) process line
    pts = sim["states"]
    fig.add_trace(go.Scatter(
        x=[p["t_db"] for p in pts], y=[p["w"] * 1000 for p in pts],
        mode="lines", line=dict(color="#ff4b4b", width=3),
        name="Actual (measured)", hoverinfo="skip"))
    colors = ["#ffd166", "#4cc9f0", "#f77f00", "#06d6a0"]
    for i, p in enumerate(pts):
        big = (selected_pt == i)
        fig.add_trace(go.Scatter(
            x=[p["t_db"]], y=[p["w"] * 1000], mode="markers+text",
            marker=dict(size=22 if big else 15, color=colors[i],
                        line=dict(color="#111" if big else "white",
                                  width=3 if big else 1.5)),
            text=[str(i + 1)], textposition="middle center",
            textfont=dict(size=11, color="#111"),
            name=f"{i+1}. {p['label']}",
            hovertemplate=(
                f"<b>{p['label']}</b> ({p['tag']})<br>"
                f"Dry bulb: {p['t_db']:.1f} deg C<br>"
                f"Humidity ratio: {p['w']*1000:.2f} g/kg<br>"
                f"RH: {p['rh']:.1f} %<br>"
                f"Enthalpy: {p['h']:.2f} kJ/kg<br>"
                f"Dew point: {p['t_dp']:.1f} deg C<br>"
                f"Wet bulb: {p['t_wb']:.1f} deg C<extra></extra>")))

    fig.add_trace(go.Scatter(
        x=[sim["t_adp"]], y=[psy.GetSatHumRatio(sim["t_adp"], P_ATM) * 1000],
        mode="markers", marker=dict(size=11, color="#ff4b4b", symbol="x-thin",
                                    line=dict(width=2.5, color="#ff4b4b")),
        name="Apparatus dew point", hoverinfo="skip"))

    fig.update_layout(
        xaxis=dict(title="Dry Bulb Temperature (deg C)", range=[0, 50],
                   gridcolor="rgba(128,128,128,0.15)", dtick=5),
        yaxis=dict(title="Humidity Ratio (g water / kg dry air)", range=[0, 27],
                   side="right", gridcolor="rgba(128,128,128,0.15)", dtick=5),
        height=460, margin=dict(l=10, r=60, t=30, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0, font=dict(size=10)),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    return fig


# ==========================================================================
# INTERACTIVE AHU SCHEMATIC  (WP1) - CHW loop, refrigerant, component highlight
# ==========================================================================

COMPONENTS = ["Intake", "Blower", "Cooling coil", "Reheater",
              "Humidifier", "Test chamber", "Chilled-water loop", "Chiller"]


def component_detail(name, sim):
    """Live, per-component explanation shown when a component is selected."""
    s1, s2, s3, s4 = sim["states"]
    if name == "Intake":
        return (f"**Intake (ST-1 / SH-1)** - {s1['t_db']:.1f} deg C, {s1['rh']:.0f}% RH, "
                f"dew point {s1['t_dp']:.1f} deg C. This fixes the moisture the coil must "
                "remove: the coil can only dehumidify if its surface falls below this "
                "dew point.")
    if name == "Blower":
        return (f"**Blower (AVE-1)** - delivering {sim['airflow_actual']:.0f} m3/h, giving "
                f"a dry-air mass flow of {sim['m_dot']:.3f} kg/s. Every load in the unit "
                "scales with this mass flow, so a filter or belt fault here weakens the "
                "coil, reheat and humidifier at once.")
    if name == "Cooling coil":
        return (f"**Cooling coil** - apparatus dew point {sim['t_adp']:.1f} deg C, bypass "
                f"factor {sim['bypass']:.2f}. Air leaves at {s2['t_db']:.1f} deg C, "
                f"{s2['rh']:.0f}% RH. "
                + (f"Currently dehumidifying at {sim['condensate']:.2f} kg/h condensate."
                   if sim["dehumidifying"] else
                   "Currently running dry - the surface is above the intake dew point, "
                   "so it cools sensibly only."))
    if name == "Reheater":
        return (f"**Electric reheater (AR-1)** - raises dry bulb from {s2['t_db']:.1f} to "
                f"{s3['t_db']:.1f} deg C at constant humidity ratio. On the chart this is a "
                "purely horizontal move to the right; RH falls as the air warms.")
    if name == "Humidifier":
        return (f"**Steam humidifier (AHUM-1)** - lifts humidity ratio from "
                f"{s3['w']*1000:.2f} to {s4['w']*1000:.2f} g/kg at nearly constant dry "
                "bulb, a near-vertical rise on the chart.")
    if name == "Test chamber":
        return (f"**Test chamber** - receives supply air at {s4['t_db']:.1f} deg C, "
                f"{s4['rh']:.0f}% RH ({s4['h']:.1f} kJ/kg). This is the conditioned space "
                "the whole process exists to serve.")
    if name == "Chilled-water loop":
        return (f"**Chilled-water loop (AB-1 pump, tank ST-15)** - carries heat from the "
                f"coil to the chiller. Supply around {sim['t_adp']-sim['approach']:.1f} deg C; "
                f"a {sim['approach']:.1f} K approach sets the coil surface temperature. "
                "Low flow here widens the approach and weakens dehumidification.")
    if name == "Chiller":
        return (f"**Chiller (A-ENF)** - a vapour-compression unit rejecting the "
                f"{sim['q_total']:.1f} kW the coil absorbs. Evaporator chills the water; "
                "compressor, condenser and expansion valve complete the refrigerant cycle.")
    return ""


def ahu_svg(sim, selected=None):
    s1, s2, s3, s4 = sim["states"]

    def glow(x, y, w, h):
        return (f'<rect x="{x-4}" y="{y-4}" width="{w+8}" height="{h+8}" rx="6" '
                f'fill="none" stroke="#ffd166" stroke-width="3" opacity="0.9"/>')

    sel = selected or ""
    hl_blower = glow(86, 104, 52, 52) if sel == "Blower" else ""
    hl_coil = glow(228, 98, 62, 64) if sel == "Cooling coil" else ""
    hl_reheat = glow(410, 98, 62, 64) if sel == "Reheater" else ""
    hl_humid = glow(580, 98, 62, 64) if sel == "Humidifier" else ""
    hl_chamber = glow(748, 72, 76, 116) if sel == "Test chamber" else ""
    hl_intake = glow(40, 95, 30, 70) if sel == "Intake" else ""
    hl_chw = (glow(250, 228, 432, 74) if sel == "Chilled-water loop" else "")
    hl_chiller = glow(690, 210, 158, 108) if sel == "Chiller" else ""

    drip = ""
    if sim["condensate"] > 0.01:
        for i, dx in enumerate([0, 9, 18]):
            drip += (
                f'<circle cx="{243+dx}" cy="164" r="2.6" fill="#4cc9f0">'
                f'<animate attributeName="cy" values="164;206" dur="1.5s" '
                f'begin="{i*0.5}s" repeatCount="indefinite"/>'
                f'<animate attributeName="opacity" values="1;1;0" dur="1.5s" '
                f'begin="{i*0.5}s" repeatCount="indefinite"/></circle>')

    speed = max(0.6, 3.0 * (500.0 / max(sim["airflow_actual"], 60)))
    coil_color = "#e63946" if sim["bypass"] > 0.25 else "#4cc9f0"

    def badge(x, s, color):
        return (
            f'<rect x="{x}" y="34" width="104" height="46" rx="5" '
            f'fill="rgba(20,24,32,0.92)" stroke="{color}" stroke-width="1.4"/>'
            f'<text x="{x+52}" y="49" font-size="9" fill="{color}" '
            f'text-anchor="middle" font-family="monospace">{s["tag"]}</text>'
            f'<text x="{x+52}" y="63" font-size="12" fill="#fff" text-anchor="middle" '
            f'font-family="monospace" font-weight="bold">{s["t_db"]:.1f} C</text>'
            f'<text x="{x+52}" y="75" font-size="10" fill="#9fb0c0" text-anchor="middle" '
            f'font-family="monospace">{s["rh"]:.0f}% RH</text>')

    coil_bars = "".join(
        f'<line x1="{234+i*9}" y1="102" x2="{234+i*9}" y2="158" '
        f'stroke="{coil_color}" stroke-width="1.6" opacity="0.65"/>' for i in range(7))
    heat_bars = "".join(
        f'<line x1="416" y1="{106+i*13}" x2="466" y2="{106+i*13}" '
        f'stroke="#e63946" stroke-width="2" opacity="0.75"/>' for i in range(5))
    steam = "".join(
        f'<circle cx="{594+i*16}" cy="{118+(i%2)*22}" r="4" fill="#06d6a0" opacity="0.7">'
        f'<animate attributeName="r" values="2;6;2" dur="2s" begin="{i*0.4}s" '
        f'repeatCount="indefinite"/></circle>' for i in range(3))

    # chilled-water pipes: coil -> chiller and back, animated flow
    chw_flow = (
        '<path d="M259,162 V240 H690" stroke="#4cc9f0" stroke-width="3" fill="none" '
        'stroke-dasharray="10 8" opacity="0.85">'
        f'<animate attributeName="stroke-dashoffset" values="36;0" dur="1.4s" '
        'repeatCount="indefinite"/></path>'
        '<path d="M690,292 H320 V240" stroke="#2a9d8f" stroke-width="3" fill="none" '
        'stroke-dasharray="10 8" opacity="0.7">'
        f'<animate attributeName="stroke-dashoffset" values="0;36" dur="1.6s" '
        'repeatCount="indefinite"/></path>')

    css = "html,body{margin:0;padding:0;background:transparent;overflow:hidden;}" \
          ".ahu-wrap{position:relative;width:100%;max-width:1100px;margin:0 auto;" \
          "aspect-ratio:860/330;}" \
          ".ahu-wrap svg{position:absolute;inset:0;width:100%;height:100%;display:block;}" \
          'text{font-family:"Source Sans Pro",system-ui,sans-serif;}'

    return f'''<!DOCTYPE html><html><head><meta charset="utf-8"><style>{css}</style></head>
    <body><div class="ahu-wrap"><svg viewBox="0 0 860 330" preserveAspectRatio="xMidYMid meet"
         xmlns="http://www.w3.org/2000/svg">
      <defs><marker id="ar" markerWidth="7" markerHeight="7" refX="6" refY="2.4"
        orient="auto"><path d="M0,0 L0,4.8 L6,2.4 z" fill="#6ee7ff"/></marker>
        <marker id="rf" markerWidth="6" markerHeight="6" refX="4" refY="2"
        orient="auto"><path d="M0,0 L0,4 L4,2 z" fill="#8a97a6"/></marker></defs>

      {hl_intake}{hl_blower}{hl_coil}{hl_reheat}{hl_humid}{hl_chamber}{hl_chw}{hl_chiller}

      <rect x="40" y="95" width="700" height="70" rx="4" fill="rgba(255,255,255,0.03)"
            stroke="#4a5568" stroke-width="1.6"/>

      <path d="M55,115 H735" stroke="#6ee7ff" stroke-width="1.6" fill="none"
            stroke-dasharray="14 10" marker-end="url(#ar)" opacity="0.75">
        <animate attributeName="stroke-dashoffset" values="48;0"
                 dur="{speed:.2f}s" repeatCount="indefinite"/></path>
      <path d="M55,145 H735" stroke="#6ee7ff" stroke-width="1.6" fill="none"
            stroke-dasharray="14 10" marker-end="url(#ar)" opacity="0.75">
        <animate attributeName="stroke-dashoffset" values="48;0"
                 dur="{speed*1.25:.2f}s" repeatCount="indefinite"/></path>

      <circle cx="112" cy="130" r="26" fill="rgba(110,231,255,0.08)"
              stroke="#6ee7ff" stroke-width="1.6"/>
      <g><path d="M112,112 L118,130 L112,148 L106,130 Z" fill="#6ee7ff" opacity="0.9"/>
        <path d="M94,130 L112,124 L130,130 L112,136 Z" fill="#6ee7ff" opacity="0.6"/>
        <circle cx="112" cy="130" r="4" fill="#6ee7ff"/>
        <animateTransform attributeName="transform" type="rotate"
          from="0 112 130" to="360 112 130" dur="{speed*0.35:.2f}s"
          repeatCount="indefinite" additive="sum"/></g>
      <text x="112" y="186" font-size="10" fill="#9fb0c0" text-anchor="middle">BLOWER AVE-1</text>
      <text x="112" y="198" font-size="9" fill="#6ee7ff" text-anchor="middle"
            font-family="monospace">{sim["airflow_actual"]:.0f} m3/h</text>

      <rect x="228" y="98" width="62" height="64" rx="3" fill="rgba(76,201,240,0.14)"
            stroke="{coil_color}" stroke-width="2"/>
      {coil_bars}
      <text x="259" y="186" font-size="10" fill="#9fb0c0" text-anchor="middle">COOLING COIL</text>
      <text x="259" y="198" font-size="9" fill="{coil_color}" text-anchor="middle"
            font-family="monospace">ADP {sim["t_adp"]:.1f} C</text>
      {drip}
      <text x="259" y="222" font-size="9" fill="#4cc9f0" text-anchor="middle"
            font-family="monospace">{sim["condensate"]:.2f} kg/h condensate</text>

      <rect x="410" y="98" width="62" height="64" rx="3" fill="rgba(230,57,70,0.14)"
            stroke="#e63946" stroke-width="2"/>
      {heat_bars}
      <text x="441" y="186" font-size="10" fill="#9fb0c0" text-anchor="middle">HEATER AR-1</text>

      <rect x="580" y="98" width="62" height="64" rx="3" fill="rgba(6,214,160,0.14)"
            stroke="#06d6a0" stroke-width="2"/>
      {steam}
      <text x="611" y="186" font-size="10" fill="#9fb0c0" text-anchor="middle">HUMIDIFIER AHUM-1</text>

      <rect x="748" y="72" width="76" height="116" rx="4" fill="rgba(255,255,255,0.04)"
            stroke="#4a5568" stroke-width="1.6"/>
      <text x="786" y="128" font-size="10" fill="#9fb0c0" text-anchor="middle">TEST</text>
      <text x="786" y="141" font-size="10" fill="#9fb0c0" text-anchor="middle">CHAMBER</text>

      <!-- chilled water circuit -->
      {chw_flow}
      <circle cx="320" cy="240" r="10" fill="rgba(76,201,240,0.15)" stroke="#4cc9f0"
              stroke-width="1.6"/>
      <path d="M315,240 L325,235 L325,245 Z" fill="#4cc9f0"/>
      <text x="320" y="266" font-size="8.5" fill="#9fb0c0" text-anchor="middle">CHW PUMP AB-1</text>
      <rect x="470" y="272" width="70" height="26" rx="3" fill="rgba(76,201,240,0.10)"
            stroke="#4cc9f0" stroke-width="1.3"/>
      <text x="505" y="288" font-size="8.5" fill="#9fb0c0" text-anchor="middle">CHW TANK ST-15</text>

      <!-- chiller: vapour-compression cycle -->
      <rect x="690" y="210" width="158" height="108" rx="5" fill="rgba(255,255,255,0.03)"
            stroke="#4a5568" stroke-width="1.6"/>
      <text x="769" y="224" font-size="8.5" fill="#c3cede" text-anchor="middle"
            font-weight="bold">CHILLER A-ENF</text>
      <text x="769" y="234" font-size="7.5" fill="#7f8c9a" text-anchor="middle">vapour compression</text>

      <!-- refrigerant loop (clockwise): COMP -> COND -> EXP -> EVAP -> COMP -->
      <g stroke="#6b7280" stroke-width="1.3" fill="none">
        <path d="M726,264 V254 H758" marker-end="url(#rf)"/>
        <path d="M780,254 H812 V264" marker-end="url(#rf)"/>
        <path d="M812,282 V292 H780" marker-end="url(#rf)"/>
        <path d="M758,292 H726 V282" marker-end="url(#rf)"/>
      </g>

      <!-- compressor (left) -->
      <circle cx="726" cy="273" r="8.5" fill="rgba(247,127,0,0.12)"
              stroke="#f77f00" stroke-width="1.6"/>
      <text x="726" y="276" font-size="6" fill="#f77f00" text-anchor="middle"
            font-family="monospace">C</text>
      <text x="704" y="276" font-size="6.5" fill="#9fb0c0" text-anchor="end">COMP</text>

      <!-- condenser (top) -->
      <rect x="758" y="248" width="22" height="12" fill="rgba(230,57,70,0.12)"
            stroke="#e63946" stroke-width="1.4"/>
      <text x="769" y="245" font-size="6.5" fill="#9fb0c0" text-anchor="middle">COND</text>

      <!-- expansion valve (right) -->
      <path d="M806,266 L806,280 L812,273 Z" fill="none" stroke="#4cc9f0" stroke-width="1.4"/>
      <path d="M818,266 L818,280 L812,273 Z" fill="none" stroke="#4cc9f0" stroke-width="1.4"/>
      <text x="833" y="276" font-size="6.5" fill="#9fb0c0" text-anchor="start">EXP</text>

      <!-- evaporator (bottom) -->
      <rect x="758" y="286" width="22" height="12" fill="rgba(6,214,160,0.12)"
            stroke="#06d6a0" stroke-width="1.4"/>
      <text x="769" y="308" font-size="6.5" fill="#9fb0c0" text-anchor="middle">EVAP</text>

      <text x="769" y="316" font-size="7.5" fill="#7f8c9a" text-anchor="middle">rejects {sim["q_total"]:.1f} kW</text>

      {badge(46, s1, "#ffd166")}
      {badge(300, s2, "#4cc9f0")}
      {badge(482, s3, "#f77f00")}
      {badge(634, s4, "#06d6a0")}
    </svg></div>
    <script>
      function _fit(){{
        var w = document.querySelector('.ahu-wrap');
        if(!w) return;
        var h = Math.ceil(w.getBoundingClientRect().height);
        try{{ if(window.frameElement) window.frameElement.style.height = h + 'px'; }}catch(e){{}}
        try{{ if(window.Streamlit) window.Streamlit.setFrameHeight(h); }}catch(e){{}}
      }}
      window.addEventListener('load', _fit);
      window.addEventListener('resize', _fit);
      new ResizeObserver(_fit).observe(document.querySelector('.ahu-wrap'));
      setTimeout(_fit, 50); setTimeout(_fit, 300);
    </script></body></html>'''


# ==========================================================================
# QUIZ  (WP1: student quiz with export)
# ==========================================================================

def build_quiz(sim):
    """Mix of concept questions and questions generated from the live state."""
    s1, s2 = sim["states"][0], sim["states"][1]
    quiz = [
        {"id": "q1", "type": "bool",
         "q": "Right now, is the cooling coil dehumidifying the air (removing moisture)?",
         "options": ["Yes", "No"],
         "answer": "Yes" if sim["dehumidifying"] else "No",
         "why": ("The coil dehumidifies only when its surface (the apparatus dew point) "
                 "is below the intake dew point.")},
        {"id": "q2", "type": "num",
         "q": "Read the current sensible heat ratio (SHR) from the coil analysis. "
              "Enter it to 2 decimals (accepted within +/-0.03).",
         "answer": round(sim["shr"], 2), "tol": 0.03,
         "why": "SHR = sensible load / total load."},
        {"id": "q3", "type": "mc",
         "q": "The electric reheater adds which kind of heat?",
         "options": ["Sensible only", "Latent only", "Both sensible and latent"],
         "answer": "Sensible only",
         "why": "Reheat raises dry bulb at constant humidity ratio - a horizontal "
                "move on the psychrometric chart."},
        {"id": "q4", "type": "mc",
         "q": "Steam humidification at constant dry bulb moves the state point which way "
              "on the psychrometric chart?",
         "options": ["Up (higher humidity ratio)", "Right (higher temperature)",
                     "Down (lower humidity ratio)"],
         "answer": "Up (higher humidity ratio)",
         "why": "Adding moisture at constant temperature raises the humidity ratio, "
                "a near-vertical rise."},
        {"id": "q5", "type": "mc",
         "q": "If the coil is running dry, which single change would start "
              "dehumidification?",
         "options": ["Lower the chilled-water temperature", "Increase the reheat duty",
                     "Increase the airflow setpoint"],
         "answer": "Lower the chilled-water temperature",
         "why": "Lowering chilled water drops the apparatus dew point below the intake "
                "dew point, so moisture begins to condense."},
    ]
    return quiz


def grade_quiz(quiz, responses):
    score, results = 0, []
    for item in quiz:
        given = responses.get(item["id"])
        if item["type"] == "num":
            try:
                ok = given is not None and abs(float(given) - item["answer"]) <= item["tol"]
            except (TypeError, ValueError):
                ok = False
        else:
            ok = (given == item["answer"])
        score += int(ok)
        results.append({"id": item["id"], "question": item["q"],
                        "your_answer": given, "correct_answer": item["answer"],
                        "correct": ok, "explanation": item["why"]})
    return score, results


# ==========================================================================
# WP2 - AI LEARNING ASSISTANT  (verified - unchanged)
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
               "generateContent" in getattr(m, "supported_generation_methods", [])]
        usable = [m for m in usable if not any(
            bad in m for bad in ("image", "video", "audio", "tts", "embedding",
                                 "veo", "imagen", "live", "native-audio"))]

        def rank(name):
            flash = "flash" in name
            lite = "lite" in name
            preview = "preview" in name or "exp" in name
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
            kw = dict(system_instruction=sys_prompt, temperature=0.4,
                      max_output_tokens=2048)
            if disable_thinking:
                kw["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
            return types.GenerateContentConfig(**kw)

        try:
            resp = client.models.generate_content(
                model=model_id, contents=contents, config=make_config(True))
        except Exception:
            resp = client.models.generate_content(
                model=model_id, contents=contents, config=make_config(False))
        text = extract_text(resp)
        if not text:
            return None, "empty"
        return text, "ok"
    except Exception as e:
        msg = str(e)
        if ("404" in msg or "limit: 0" in msg) and "model_options" in st.session_state:
            opts = st.session_state.model_options
            if model_id in opts and opts.index(model_id) + 1 < len(opts):
                st.session_state.model_id = opts[opts.index(model_id) + 1]
                return ask_llm(question, sim, controls, history)
        st.session_state.last_api_error = msg
        return None, "unavailable"


def offline_answer(question, sim, controls):
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
# DATA INGESTION LAYER  (real-time sensor analytics scaffold)
# ==========================================================================
# The whole app reads from one `sim` dict of the shape simulate() returns. That
# dict is produced from a *data source*, not from the sliders directly, so the
# source can be swapped without touching the chart, tutor, diagnostics or
# schematic. Two sources are provided:
#
#   SimSource  - the digital twin driven by the sidebar sliders (default).
#   LiveSource - reads tagged sensor rows from a CSV feed in the rig's export
#                format. Point it at the EDIBON data-management CSV (or later an
#                MQTT / OPC-UA / nidaqmx bridge) and nothing downstream changes.
#
# In Live mode the four state points are built from *measured* sensor readings,
# while simulate() runs alongside as the *theoretical* reference the chart
# overlays them against - so "measured vs theoretical" becomes literally real.

SENSOR_TAGS = ["ST-1", "SH-1", "ST-5", "SH-3", "ST-7", "SH-4",
               "ST-9", "SH-5", "SC-1", "ST-13"]
_APP_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() \
    else os.getcwd()
DEFAULT_FEED = os.path.join(_APP_DIR, "sample_sensor_feed.csv")


class DataSource:
    """Any source that can yield one frame of tagged sensor readings."""
    def read(self):
        raise NotImplementedError


class SimSource(DataSource):
    """Emit the slider-driven twin's state as tagged readings (transparency)."""
    def __init__(self, sim, chw):
        self.sim = sim
        self.chw = chw

    def read(self):
        s = self.sim["states"]
        return {
            "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
            "source": "sim", "row": None, "rows": None,
            "ST-1": round(s[0]["t_db"], 2), "SH-1": round(s[0]["rh"], 1),
            "ST-5": round(s[1]["t_db"], 2), "SH-3": round(s[1]["rh"], 1),
            "ST-7": round(s[2]["t_db"], 2), "SH-4": round(s[2]["rh"], 1),
            "ST-9": round(s[3]["t_db"], 2), "SH-5": round(s[3]["rh"], 1),
            "SC-1": round(self.sim["airflow_actual"], 1), "ST-13": round(self.chw, 2),
        }


class LiveSource(DataSource):
    """Read one frame from a CSV feed in the rig's sensor-tag format.

    Swapping this file for the EDIBON data-management export - or replacing the
    file read with an MQTT subscribe / OPC-UA read / nidaqmx sample - is the
    only change needed to go from mock feed to real rig.
    """
    def __init__(self, path, row_index=None):
        self.path = path
        self.row_index = row_index          # None -> latest row

    def read(self):
        with open(self.path, newline="") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            return None
        idx = len(rows) - 1 if self.row_index is None else self.row_index % len(rows)
        raw = rows[idx]
        frame = {"source": "live", "row": idx + 1, "rows": len(rows),
                 "timestamp": raw.get("timestamp", "")}
        for tag in SENSOR_TAGS:
            if tag in raw and raw[tag] not in ("", None):
                frame[tag] = float(raw[tag])
        missing = [t for t in SENSOR_TAGS if t not in frame]
        if missing:
            raise ValueError(f"feed missing sensor tags: {', '.join(missing)}")
        return frame


def states_from_readings(r):
    """Build the four psychrometric state points from measured T + RH pairs."""
    def st_pt(t, rh, label, tag):
        w = psy.GetHumRatioFromRelHum(t, max(min(rh, 100.0), 1.0) / 100.0, P_ATM)
        return state(t, w, label, tag)
    return [
        st_pt(r["ST-1"], r["SH-1"], "Intake air", "ST-1 / SH-1"),
        st_pt(r["ST-5"], r["SH-3"], "After cooling coil", "ST-5 / SH-3"),
        st_pt(r["ST-7"], r["SH-4"], "After reheat", "ST-7 / SH-4"),
        st_pt(r["ST-9"], r["SH-5"], "Supply to chamber", "ST-9 / SH-5"),
    ]


def sim_from_readings(r):
    """Assemble a simulate()-shaped dict from measured readings, so every
    downstream consumer (KPIs, diagnostics, chart, tutor, schematic) works
    unchanged. Coil bypass and ADP are *inferred* from the measured air temps,
    since the coil surface itself is not directly sensed."""
    s = states_from_readings(r)
    s1, s2, s3 = s[0], s[1], s[2]
    airflow = r.get("SC-1", 500.0)
    chw = r.get("ST-13", 7.0)
    m_dot = (airflow / 3600.0) / s1["v"]
    cp_moist = CP_AIR + CP_VAP * s2["w"]
    q_total = m_dot * (s1["h"] - s2["h"])
    q_sens = m_dot * cp_moist * (s1["t_db"] - s2["t_db"])
    q_lat = q_total - q_sens
    shr = q_sens / q_total if abs(q_total) > 1e-6 else 0.0
    condensate = max(m_dot * (s1["w"] - s2["w"]) * 3600.0, 0.0)
    approach = 1.5                                   # nominal, chilled-water side
    t_adp = chw + approach
    denom = s1["t_db"] - t_adp
    bypass = (s2["t_db"] - t_adp) / denom if abs(denom) > 0.2 else 0.12
    bypass = min(max(bypass, 0.0), 1.0)              # inferred effective bypass
    return {
        "states": s, "m_dot": m_dot, "airflow_actual": airflow,
        "t_adp": t_adp, "bypass": bypass, "approach": approach,
        "q_total": q_total, "q_sens": q_sens, "q_lat": q_lat, "shr": shr,
        "condensate": condensate,
        "dehumidifying": (s1["w"] - s2["w"]) > 1e-5,
        "heater_tripped": False, "dt_reheat_demand": s3["t_db"] - s2["t_db"],
        "faults": [], "measured": True,
    }


# ==========================================================================
# GUIDED LEARNING SCENARIOS  (WP1: self-directed, task-based learning)
# ==========================================================================
# Each scenario presets the rig to a starting condition (slider values, and
# optionally a silently-injected fault), frames a task, walks the student
# through what to do and observe, and self-checks understanding. This is the
# self-directed student mode.

SCENARIOS = [
    {
        "id": "dehumidify",
        "title": "Humid day - hit the dehumidification target",
        "situation": "A hot, humid day: intake air is 32 deg C at 75% RH and the "
                     "chilled water is running warm at 14 deg C. The coil is only lightly "
                     "dehumidifying and the supply air is still too moist for the chamber. "
                     "Your job is to increase the dehumidification.",
        "objective": "Understand how the chilled-water temperature sets the apparatus "
                     "dew point, and how lowering it pushes the coil surface further "
                     "below the intake dew point so more moisture condenses out.",
        "controls": {"t_intake": 32.0, "rh_intake": 75.0, "airflow": 500,
                     "t_chw": 14.0, "reheat_kw": 1.0, "humid_kgh": 0.0},
        "faults": [],
        "steps": [
            "Open the WP3 tab. Read the coil outlet point (2) - some moisture is being "
            "removed, but is it enough? Note the condensate rate on the schematic.",
            "Back in the sidebar, lower the Chilled Water Temp slider from 14 toward "
            "7 deg C and watch the apparatus dew point (ADP) drop further below the "
            "intake dew point.",
            "Watch the condensate rate climb, the supply humidity ratio fall further, "
            "and the measured process on the chart bend more steeply downward.",
        ],
        "check": {
            "q": "What increased the dehumidification?",
            "options": ["Lowering the chilled-water temperature",
                        "Increasing the reheat duty",
                        "Increasing the airflow"],
            "answer": "Lowering the chilled-water temperature",
            "why": "It dropped the apparatus dew point further below the intake dew "
                   "point, so more moisture condenses on the coil.",
            "hint": "Look at what has to happen to the coil surface temperature for "
                    "more water to condense out of the air.",
        },
        "explanation": "At 14 deg C chilled water the apparatus dew point is only a "
                       "little below the intake dew point, so the coil removes just a "
                       "modest amount of moisture. As you lower the chilled water, the "
                       "ADP drops further below the intake dew point, the condensate "
                       "rate rises, the humidity ratio falls further, and the coil load "
                       "shifts toward latent (the SHR drops).",
    },
    {
        "id": "coil_fault",
        "title": "Fault-finding - why is the supply air warm?",
        "situation": "Students report the chamber isn't cooling properly. The setpoints "
                     "look normal - 32 deg C / 70% intake, 7 deg C chilled water - but "
                     "something in the plant is off. Diagnose it from the symptoms, using "
                     "the schematic, the chart and the diagnostics. You are not told what "
                     "the fault is.",
        "objective": "Practise diagnosing a coil fault from its signature: a warm coil "
                     "outlet, a raised bypass factor, and a measured process that has "
                     "pulled away from the theoretical clean plant.",
        "controls": {"t_intake": 32.0, "rh_intake": 70.0, "airflow": 500,
                     "t_chw": 7.0, "reheat_kw": 1.5, "humid_kgh": 0.5},
        "faults": ["Fouled cooling coil"],
        "steps": [
            "Look at the cooling coil on the WP1 schematic - what colour is it, and what "
            "does that colour indicate?",
            "Read the diagnostics banner and the coil-outlet temperature (point 2). Is "
            "the air leaving the coil colder or warmer than a healthy coil would give?",
            "Open the WP3 tab - how far has the red measured process separated from the "
            "dashed green theoretical clean-plant process?",
        ],
        "check": {
            "q": "What is the most likely fault?",
            "options": ["Fouled cooling coil", "Humidifier stuck on", "Low airflow"],
            "answer": "Fouled cooling coil",
            "why": "The coil bypass factor has risen from 0.12 to about 0.38, so more "
                   "air slips through untreated and leaves warmer and wetter.",
            "hint": "A red coil, a warm coil outlet, and a big measured-vs-theoretical "
                    "gap all point at the coil itself, not the fan or humidifier.",
        },
        "explanation": "A fouled coil raises the bypass factor from 0.12 to ~0.38: a "
                       "larger fraction of air passes through without touching the cold "
                       "fins, so the coil outlet is warmer and drier than it should be. "
                       "The schematic shows the coil in red, the diagnostics flag coil "
                       "underperformance, and on the chart the measured process sits well "
                       "away from the theoretical clean-plant line.",
    },
]
SCENARIOS_BY_ID = {s["id"]: s for s in SCENARIOS}   # (kept for reference/lookup)


def apply_scenario(scn):
    """Preset the sliders to the scenario's starting conditions (once per switch)."""
    for key, val in scn["controls"].items():
        st.session_state[key] = val
    st.session_state["applied_scenario"] = scn["id"]


# ==========================================================================
# UI
# ==========================================================================

st.title("AI-Enabled Intelligent HVAC Learning Platform")
st.caption("Real-Time Digital Twin | ASHRAE Psychrometrics | AI Assistant & Diagnostics")

with st.sidebar:
    st.header("Learning Mode")
    mode = st.radio(
        "Mode", ["Guided walkthrough", "Guided scenarios", "Instructor demonstration"],
        label_visibility="collapsed",
        help="Guided scenarios is the self-directed student mode. Instructor mode "
             "unlocks fault injection and quiz answer keys.")
    instructor = mode == "Instructor demonstration"

    scenario = None
    if mode == "Guided scenarios":
        st.caption("Self-directed scenarios - pick one to set up the rig.")
        _titles = [s["title"] for s in SCENARIOS]
        _pick = st.selectbox("Scenario", _titles, key="scn_pick",
                             label_visibility="collapsed")
        scenario = SCENARIOS[_titles.index(_pick)]
        # apply presets once, when the selection changes, so manual slider
        # adjustments afterwards are not overwritten on every rerun
        if st.session_state.get("applied_scenario") != scenario["id"]:
            apply_scenario(scenario)
    else:
        st.session_state["applied_scenario"] = None

    st.divider()
    st.header("Data Source")
    data_mode = st.radio(
        "Data source", ["Simulated (sliders)", "Live ingestion (sensor feed)"],
        label_visibility="collapsed",
        help="Live mode builds the measured state points from a CSV sensor feed "
             "in the rig's export format. Swap the file for the EDIBON export "
             "(or an MQTT/OPC-UA bridge) with no other change.")
    live_mode = data_mode.startswith("Live")
    feed_path = DEFAULT_FEED
    if live_mode:
        st.session_state.setdefault("feed_row", 0)
        feed_path = st.text_input("Sensor feed CSV path", DEFAULT_FEED)
        fc1, fc2 = st.columns(2)
        if fc1.button("Advance feed", use_container_width=True):
            st.session_state.feed_row = st.session_state.get("feed_row", 0) + 1
        if fc2.button("Reset feed", use_container_width=True):
            st.session_state.feed_row = 0
        st.caption("Replays the mock feed row by row so the measured line moves "
                   "as you step through it. A real feed streams new rows live.")

    st.divider()
    st.header("Live Sensor Controls")
    if live_mode:
        st.caption("Intake conditions come from the feed in Live mode; these "
                   "sliders drive the theoretical reference model.")
    else:
        st.caption("Intake conditions")
    t_intake = st.slider("Intake Dry Bulb (deg C)", 15.0, 45.0, 32.0, 0.5, key="t_intake")
    rh_intake = st.slider("Intake Relative Humidity (%)", 20.0, 95.0, 70.0, 1.0, key="rh_intake")
    airflow = st.slider("Airflow Setpoint (m3/h)", 100, 1000, 500, 25, key="airflow")

    st.caption("Plant setpoints")
    t_chw = st.slider("Chilled Water Temp (deg C)", 4.0, 20.0, 7.0, 0.5, key="t_chw")
    reheat_kw = st.slider("Reheat Duty (kW)", 0.0, 5.0, 1.5, 0.1, key="reheat_kw")
    humid_kgh = st.slider("Humidifier Output (kg/h)", 0.0, 6.0, 0.5, 0.1, key="humid_kgh")

    faults, sensor_faults = [], []
    if instructor:
        st.divider()
        st.subheader("Plant Fault Injection")
        st.caption("Degrade the physical plant and see if students catch it.")
        faults = st.multiselect(
            "Plant fault", ["Fouled cooling coil", "Clogged air filter",
                            "Fan belt slipping", "Low chilled water flow"],
            label_visibility="collapsed")
        st.subheader("Sensor Fault Injection")
        st.caption("Corrupt a reading without touching the physics. Caught by "
                   "physical-consistency checks, not by magnitude.")
        sensor_faults = st.multiselect(
            "Sensor fault", list(SENSOR_FAULTS.keys()), label_visibility="collapsed")
    else:
        st.divider()
        st.caption("Fault injection is available in Instructor demonstration mode.")

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

# ---- resolve the active data source --------------------------------------
feed_error = None
readings = None
if live_mode:
    try:
        src = LiveSource(feed_path, row_index=st.session_state.get("feed_row"))
        readings = src.read()
        sim = sim_from_readings(readings)
        # theoretical reference: clean plant on the *measured* intake + setpoints
        sim_ideal = simulate(readings["ST-1"], readings["SH-1"], readings["SC-1"],
                             readings["ST-13"], reheat_kw, humid_kgh, [])
        airflow_ref = readings["SC-1"]
        faults, sensor_faults = [], []          # detected, not injected, in Live
        controls = {
            "intake_dry_bulb_C": round(readings["ST-1"], 1),
            "intake_RH_pct": round(readings["SH-1"], 1),
            "airflow_setpoint": round(readings["SC-1"], 0),
            "chilled_water_C": round(readings["ST-13"], 1),
            "reheat_kW": reheat_kw, "humidifier_kg_h": humid_kgh,
            "data_source": "live sensor feed",
        }
    except Exception as e:
        feed_error = str(e)
        live_mode = False                        # graceful fallback to simulated

if not live_mode:
    scenario_faults = scenario["faults"] if (mode == "Guided scenarios" and scenario) else []
    faults = list(dict.fromkeys(faults + scenario_faults))   # merge, dedupe
    controls = {
        "intake_dry_bulb_C": t_intake, "intake_RH_pct": rh_intake,
        "airflow_setpoint": airflow, "chilled_water_C": t_chw,
        "reheat_kW": reheat_kw, "humidifier_kg_h": humid_kgh,
        "data_source": "simulated (sliders)",
    }
    sim = simulate(t_intake, rh_intake, airflow, t_chw, reheat_kw, humid_kgh, faults)
    sim_ideal = simulate(t_intake, rh_intake, airflow, t_chw, reheat_kw, humid_kgh, [])
    airflow_ref = airflow
    readings = SimSource(sim, t_chw).read()

findings = diagnose(sim, airflow_ref)
reported, corrupted = apply_sensor_faults(sim, sensor_faults)
sensor_findings = diagnose_sensors(sim, reported)

if feed_error:
    st.error(f"Live feed unavailable ({feed_error}). Fell back to simulated data.")

# ---- diagnostics banner --------------------------------------------------
st.subheader("Predictive Diagnostics & Anomaly Detection")
for lvl, title, detail in findings:
    {"error": st.error, "warn": st.warning, "ok": st.success}[lvl](f"**{title}** - {detail}")
if instructor or sensor_faults:
    for lvl, title, detail in sensor_findings:
        {"error": st.error, "warn": st.warning, "ok": st.success}[lvl](f"**{title}** - {detail}")

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Supply Air", f"{sim['states'][3]['t_db']:.1f} C",
          f"{sim['states'][3]['t_db'] - t_intake:+.1f} C vs intake")
k2.metric("Supply RH", f"{sim['states'][3]['rh']:.0f} %")
k3.metric("Coil Load", f"{sim['q_total']:.2f} kW", f"SHR {sim['shr']:.2f}",
          delta_color="off")
k4.metric("Condensate", f"{sim['condensate']:.2f} kg/h")
k5.metric("Mass Flow", f"{sim['m_dot']:.3f} kg/s")

st.divider()

tab1, tab2, tab3, tab_quiz = st.tabs([
    "WP1 - Immersive AHU Environment",
    "WP2 - AI Learning Assistant",
    "WP3 - Psychrometric Visualization",
    "Student Quiz"])

# ---- WP3 ------------------------------------------------------------------
with tab3:
    with st.expander(
            ("LIVE" if live_mode else "SIMULATED") + " data source - "
            + ("streaming from sensor feed" if live_mode else "digital twin (sliders)"),
            expanded=live_mode):
        if live_mode:
            st.caption(f"Source: {readings['source']} | frame {readings['row']} of "
                       f"{readings['rows']} | timestamp {readings['timestamp']}")
            st.caption("Measured state points below are built from these raw sensor "
                       "readings through the same ingestion pipeline the rig would use. "
                       "Swap this file for the EDIBON export to go fully live.")
        else:
            st.caption("Running on the digital twin. Switch to Live ingestion in the "
                       "sidebar to feed measured state points from a sensor CSV.")
        st.dataframe([{
            "ST-1": readings.get("ST-1"), "SH-1": readings.get("SH-1"),
            "ST-5": readings.get("ST-5"), "SH-3": readings.get("SH-3"),
            "ST-7": readings.get("ST-7"), "SH-4": readings.get("SH-4"),
            "ST-9": readings.get("ST-9"), "SH-5": readings.get("SH-5"),
            "SC-1": readings.get("SC-1"), "ST-13": readings.get("ST-13"),
        }], hide_index=True, use_container_width=True)

    c1, c2 = st.columns([3, 2])
    with c1:
        st.plotly_chart(psych_chart(sim, sim_ideal), use_container_width=True)
        st.caption("Red = actual measured process. Dashed green = theoretical "
                   "clean-plant process for the same setpoints. When they diverge, "
                   "the plant is deviating from ideal - inject a fault to see it open up.")

        st.markdown("**AI decision support - operation & optimisation**")
        for pri, text in recommend(sim, controls):
            {"high": st.error, "med": st.warning, "ok": st.success}.get(
                pri, st.info)(text)
    with c2:
        st.markdown("**State point properties**")
        if instructor or sensor_faults:
            st.caption("Values shown are the *reported* readings. Cells flagged with a "
                       "warning are corrupted by an injected sensor fault.")
        rows = []
        for i, s in enumerate(reported):
            flag = " (!)" if corrupted.get(i) else ""
            rows.append({
                "Pt": i + 1, "Location": s["label"], "Tag": s["tag"] + flag,
                "T db (C)": round(s["t_db"], 1), "RH (%)": round(s["rh"], 1),
                "W (g/kg)": round(s["w"] * 1000, 2), "h (kJ/kg)": round(s["h"], 2),
                "T dp (C)": round(s["t_dp"], 1), "T wb (C)": round(s["t_wb"], 1),
                "v (m3/kg)": round(s["v"], 4)})
        st.dataframe(rows, hide_index=True, use_container_width=True)

        st.markdown("**Cooling coil heat transfer**")
        st.dataframe([
            {"Quantity": "Total (latent + sensible)", "Value": f"{sim['q_total']:.3f} kW"},
            {"Quantity": "Sensible", "Value": f"{sim['q_sens']:.3f} kW"},
            {"Quantity": "Latent", "Value": f"{sim['q_lat']:.3f} kW"},
            {"Quantity": "Sensible heat ratio", "Value": f"{sim['shr']:.3f}"},
            {"Quantity": "Apparatus dew point", "Value": f"{sim['t_adp']:.2f} C"},
            {"Quantity": "Bypass factor", "Value": f"{sim['bypass']:.3f}"},
        ], hide_index=True, use_container_width=True)

        # CSV export of state points
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(["point", "location", "tag", "t_db_C", "rh_pct", "w_g_kg",
                    "h_kJ_kg", "t_dp_C", "t_wb_C", "v_m3_kg"])
        for i, s in enumerate(sim["states"]):
            w.writerow([i + 1, s["label"], s["tag"], round(s["t_db"], 2),
                        round(s["rh"], 1), round(s["w"] * 1000, 3), round(s["h"], 2),
                        round(s["t_dp"], 2), round(s["t_wb"], 2), round(s["v"], 4)])
        e1, e2 = st.columns(2)
        e1.download_button("Export state points (CSV)", buf.getvalue(),
                           file_name="hvac_state_points.csv", mime="text/csv",
                           use_container_width=True)
        e2.download_button("Export analysis (JSON)", build_context(sim, controls),
                           file_name="hvac_analysis.json", mime="application/json",
                           use_container_width=True)
        st.caption("Charts export to PNG from the camera icon on the chart toolbar.")

# ---- WP1 ------------------------------------------------------------------
with tab1:
    st.markdown("**Live AHU cutaway** - airflow, coil condensate, the chilled-water "
                "loop and the chiller's vapour-compression cycle are all driven by the "
                "same simulation core as the chart.")

    selected = st.selectbox("Select a component to inspect", COMPONENTS, index=2)
    components.html(ahu_svg(sim, selected), height=470, scrolling=True)
    st.info(component_detail(selected, sim))
    st.caption("Streamline animation rate scales with delivered airflow. Droplets appear "
               "only when the coil surface falls below the intake dew point. The coil "
               "outline turns red when its bypass factor degrades. Chilled-water flow "
               "animates from coil to chiller and back.")

    if mode == "Guided walkthrough":
        st.markdown("### Guided walkthrough")
        steps = [
            ("Intake", "Ambient air is drawn in past ST-1/SH-1. This fixes the intake "
                       "dew point, which decides whether the coil can dehumidify."),
            ("Cooling coil", "The coil cools air toward its apparatus dew point. Air "
                             "that contacts the fins leaves saturated at ADP; the rest "
                             "bypasses untreated. Mixing gives the outlet state."),
            ("Reheater", "The electric reheater adds sensible heat only - a horizontal "
                         "move right on the chart. Humidity ratio is unchanged, RH falls."),
            ("Humidifier", "The humidifier injects steam, raising humidity ratio at "
                           "nearly constant dry bulb - a near-vertical rise."),
        ]
        for name, txt in steps:
            with st.expander(f"Step: {name}"):
                st.markdown(component_detail(name, sim))
                st.markdown(txt)
    elif mode == "Guided scenarios":
        st.markdown("### Guided learning scenarios")
        st.caption("Self-directed learning: choose a scenario from the sidebar and the "
                   "rig is set to its starting condition automatically. Then work through "
                   "the steps below, adjusting the sliders as directed.")
        if live_mode:
            st.warning("You are in Live ingestion mode. Switch Data Source to "
                       "Simulated so the scenario's conditions drive the rig.")
        scn = scenario
        st.markdown(f"**Scenario.** {scn['title']}")
        st.markdown(f"**Situation.** {scn['situation']}")
        st.markdown(f"**Learning objective.** {scn['objective']}")
        st.markdown("**Steps**")
        for i, step_txt in enumerate(scn["steps"], 1):
            st.markdown(f"{i}. {step_txt}")

        chk = scn["check"]
        ans = st.radio(chk["q"], chk["options"], index=None, key=f"scn_check_{scn['id']}")
        if ans is not None:
            if ans == chk["answer"]:
                st.success(f"Correct. {chk['why']}")
            else:
                st.info(f"Not quite - {chk['hint']}")
        with st.expander("Explanation - what you should see"):
            st.markdown(scn["explanation"])
    elif mode == "Instructor demonstration":
        st.markdown("### Instructor notes")
        st.markdown("Inject a plant or sensor fault from the sidebar and ask students to "
                    "read the schematic and diagnostics before you reveal the cause. The "
                    "coil turns red for a fouled coil; the chart's actual vs theoretical "
                    "paths separate; sensor faults surface as physical-consistency errors.")

# ---- WP2 (student quiz lives here too) ------------------------------------
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

    box = st.container(height=320)
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

# ---- Student Quiz (own tab) ----------------------------------------------
with tab_quiz:
    st.markdown("### Student Quiz")
    st.caption("Questions marked (live) are generated from the current rig state, so the "
               "answer changes as you adjust the plant. Answer keys appear in "
               "Instructor demonstration mode.")
    quiz = build_quiz(sim)
    if "quiz_responses" not in st.session_state:
        st.session_state.quiz_responses = {}

    name = st.text_input("Student name (for the export record)", "")
    responses = {}
    for n, item in enumerate(quiz, 1):
        live = " (live)" if item["id"] in ("q1", "q2", "q5") else ""
        if item["type"] == "num":
            responses[item["id"]] = st.text_input(f"Q{n}{live}. {item['q']}", key=item["id"])
        else:
            responses[item["id"]] = st.radio(
                f"Q{n}{live}. {item['q']}", item["options"], key=item["id"], index=None)
        if instructor:
            st.caption(f"Answer key: {item['answer']} - {item['why']}")

    if st.button("Submit quiz"):
        score, results = grade_quiz(quiz, responses)
        st.success(f"Score: {score} / {len(quiz)}")
        for r in results:
            (st.success if r["correct"] else st.error)(
                f"{'Correct' if r['correct'] else 'Review'}: {r['question']}  \n"
                f"Your answer: {r['your_answer']} | Correct: {r['correct_answer']}  \n"
                f"{r['explanation']}")
        record = {
            "student": name or "anonymous",
            "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
            "mode": mode,
            "rig_state": json.loads(build_context(sim, controls)),
            "score": score, "out_of": len(quiz), "results": results,
        }
        st.download_button(
            "Export quiz result (JSON)", json.dumps(record, indent=2),
            file_name="hvac_quiz_result.json", mime="application/json")
        st.caption("Export target: local workstation download, or wire this JSON to an "
                   "institutional LMS / cloud endpoint.")
