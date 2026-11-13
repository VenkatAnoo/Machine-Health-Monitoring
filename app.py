import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

app = FastAPI(title="AI4I Predictive Maintenance API")
templates = Jinja2Templates(directory="templates")

# Load trained model pipeline
try:
  model = joblib.load("final_torque_prediction_model.pkl")
  model_loaded = True
except Exception:
  model = None
  model_loaded = False


class TelemetryInput(BaseModel):
  air_temp: float
  process_temp: float
  rotational_speed: int
  tool_wear: int
  machine_type: str


@app.get("/", response_class=HTMLResponse)
async def serve_dashboard(request: Request):
  return templates.TemplateResponse(
      request=request,
      name="index.html",
      context={"model_loaded": model_loaded},
  )


@app.post("/predict")
async def predict_torque(data: TelemetryInput):
  if model is None:
    raise HTTPException(
        status_code=500, detail="Model file not loaded successfully."
    )

  input_df = pd.DataFrame({
      "Air temperature [K]": [data.air_temp],
      "Process temperature [K]": [data.process_temp],
      "Rotational speed [rpm]": [data.rotational_speed],
      "Tool wear [min]": [data.tool_wear],
      "Type": [data.machine_type],
  })

  predicted_torque = float(model.predict(input_df)[0])

  # Physics indicators (Fixed syntax: 2 * np.pi)
  angular_velocity = data.rotational_speed * (2 * np.pi / 60)
  mechanical_power_kw = (predicted_torque * angular_velocity) / 1000.0
  temp_diff = data.process_temp - data.air_temp
  overstrain_metric = data.tool_wear * predicted_torque

  os_limit = {"L": 11000, "M": 12000, "H": 13000}.get(data.machine_type, 12000)

  alerts = []
  if predicted_torque > 60.0:
    alerts.append({
        "level": "critical",
        "title": "CRITICAL TORQUE LOAD",
        "msg": "Torque exceeds 60 Nm. Risk of spindle stall.",
    })
  elif predicted_torque > 48.0:
    alerts.append({
        "level": "warning",
        "title": "ELEVATED TORQUE",
        "msg": "Operating in upper load band (48–60 Nm).",
    })

  if mechanical_power_kw < 3.5 or mechanical_power_kw > 9.0:
    alerts.append({
        "level": "warning",
        "title": "Power Failure (PWF) Risk",
        "msg": f"Calculated power ({mechanical_power_kw:.2f} kW) out of bounds.",
    })

  if temp_diff < 8.6 and data.rotational_speed < 1380:
    alerts.append({
        "level": "warning",
        "title": "Heat Dissipation (HDF) Risk",
        "msg": f"ΔT ({temp_diff:.1f} K) < 8.6 K at low RPM.",
    })

  if overstrain_metric > os_limit:
    alerts.append({
        "level": "critical",
        "title": "Overstrain Failure (OSF) Risk",
        "msg": f"Strain index ({overstrain_metric:,.0f}) exceeds limit.",
    })

  if not alerts:
    alerts.append({
        "level": "ok",
        "title": "NOMINAL OPERATION",
        "msg": "All telemetry parameters are within safe bounds.",
    })

  return {
      "torque": round(predicted_torque, 2),
      "power_kw": round(mechanical_power_kw, 2),
      "temp_diff": round(temp_diff, 1),
      "overstrain": round(overstrain_metric, 0),
      "alerts": alerts,
  }