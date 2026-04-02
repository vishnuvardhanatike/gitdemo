import streamlit as st
import requests
import plotly.graph_objects as go

st.set_page_config(layout="wide")

st.title("AI-Based Power Pole Failure Prediction")

# Weather Inputs
rainfall = st.slider("Rainfall (mm)", 0, 200, 50)
wind_speed = st.slider("Wind Speed (km/h)", 0, 150, 40)
humidity = st.slider("Humidity (%)", 0, 100, 70)
temperature = st.slider("Temperature (°C)", 0, 50, 30)

if st.button("Run Prediction"):

    try:
        params = {
            "rainfall": rainfall,
            "wind_speed": wind_speed,
            "humidity": humidity,
            "temperature": temperature
        }

        response = requests.get(
            "http://127.0.0.1:8000/predict",
            params=params,
            timeout=20
        )

        if response.status_code != 200:
            st.error(f"Backend Error: {response.status_code}")
            st.stop()

        data = response.json()

    except requests.exceptions.RequestException as e:
        st.error("Backend not responding.")
        st.stop()

    poles = data["poles"]
    connections = data["connections"]

    fig = go.Figure()

    # Draw connections
    for conn in connections:
        p1 = next(p for p in poles if p["id"] == conn[0])
        p2 = next(p for p in poles if p["id"] == conn[1])

        fig.add_trace(go.Scattermapbox(
            lon=[p1["lon"], p2["lon"]],
            lat=[p1["lat"], p2["lat"]],
            mode="lines",
            line=dict(width=1),
            showlegend=False
        ))

    # Draw poles
    fig.add_trace(go.Scattermapbox(
        lon=[p["lon"] for p in poles],
        lat=[p["lat"] for p in poles],
        mode="markers",
        marker=dict(
            size=8,
            color=[
                "red" if p["failed"] else "green"
                for p in poles
            ]
        ),
        text=[
            f"Failure Prob: {round(p['failure_probability'], 2)}"
            for p in poles
        ]
    ))

    fig.update_layout(
        mapbox=dict(
            style="open-street-map",
            zoom=16,
            center=dict(
                lat=poles[0]["lat"],
                lon=poles[0]["lon"]
            )
        ),
        margin=dict(l=0, r=0, t=0, b=0)
    )

    st.plotly_chart(fig, use_container_width=True)