import { useEffect, useState } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";
import "./App.css";

// API endpoint for the FastAPI backend.
const API_URL = "http://127.0.0.1:8001";

// Default sensor values used for the initial dashboard.
const initialSensors = {
  air_temperature: 298.5,
  process_temperature: 308.6,
  rotational_speed: 1500,
  torque: 40,
  tool_wear: 100,
};

function App() {
  const [equipmentId, setEquipmentId] = useState("MACHINE-001");
  const [sensors, setSensors] = useState(initialSensors);
  const [prediction, setPrediction] = useState(null);
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  // Convert Kelvin to Celsius for dashboard display.
  const toCelsius = (kelvin) => (kelvin - 273.15).toFixed(1);

  // Fetch prediction history from FastAPI.
  const fetchHistory = async () => {
    try {
      const response = await fetch(`${API_URL}/predictions`);

      if (!response.ok) {
        throw new Error("Could not load prediction history.");
      }

      const data = await response.json();

      setHistory(
        data.map((item, index) => ({
          id: index,
          risk: Number((item.failure_probability * 100).toFixed(1)),
        }))
      );
    } catch (err) {
      setError(err.message);
    }
  };

  // Send current sensor values to FastAPI.
  const runPrediction = async () => {
    setLoading(true);
    setError("");

    try {
      const response = await fetch(`${API_URL}/predict`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          equipment_id: equipmentId,
          ...sensors,
          type: "L",
        }),
      });

      if (!response.ok) {
        throw new Error("Prediction request failed.");
      }

      const data = await response.json();

      setPrediction(data);

      // Refresh prediction history after a new prediction.
      await fetchHistory();
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  // Load existing history when dashboard opens.
  useEffect(() => {
    fetchHistory();
  }, []);

  const risk = prediction
    ? Number((prediction.failure_probability * 100).toFixed(1))
    : 0;

  const status = prediction?.maintenance_status || "Awaiting prediction";

  return (
    <div className="app">
      <header className="header">
        <div>
          <h1>Vehicle Health Monitor</h1>
          <p>Predictive Maintenance & Telemetry Analytics</p>
        </div>

        <div className="equipment">
          <label>Equipment</label>

          <input
            value={equipmentId}
            onChange={(event) => setEquipmentId(event.target.value)}
          />
        </div>
      </header>

      <main className="dashboard">
        <section className="hero-card">
          <div>
            <span className="eyebrow">CURRENT HEALTH</span>

            <h2>
              {prediction
                ? prediction.failure_prediction
                  ? "Attention Required"
                  : "Operating Normally"
                : "No Prediction Yet"}
            </h2>

            <p>{status}</p>
          </div>

          <div className="risk">
            <span>Failure Risk</span>
            <strong>{risk}%</strong>
          </div>
        </section>

        <section className="sensor-grid">
          <SensorCard
            title="Air Temperature"
            value={`${toCelsius(sensors.air_temperature)}°C`}
          />

          <SensorCard
            title="Process Temperature"
            value={`${toCelsius(sensors.process_temperature)}°C`}
          />

          <SensorCard
            title="Rotational Speed"
            value={`${sensors.rotational_speed} RPM`}
          />

          <SensorCard
            title="Torque"
            value={`${sensors.torque} Nm`}
          />

          <SensorCard
            title="Tool Wear"
            value={`${sensors.tool_wear} min`}
          />
        </section>

        <section className="controls">
          <h2>Sensor Input</h2>

          <div className="input-grid">
            <SensorInput
              label="Air Temperature (K)"
              value={sensors.air_temperature}
              onChange={(value) =>
                setSensors({ ...sensors, air_temperature: value })
              }
            />

            <SensorInput
              label="Process Temperature (K)"
              value={sensors.process_temperature}
              onChange={(value) =>
                setSensors({ ...sensors, process_temperature: value })
              }
            />

            <SensorInput
              label="RPM"
              value={sensors.rotational_speed}
              onChange={(value) =>
                setSensors({ ...sensors, rotational_speed: value })
              }
            />

            <SensorInput
              label="Torque (Nm)"
              value={sensors.torque}
              onChange={(value) =>
                setSensors({ ...sensors, torque: value })
              }
            />

            <SensorInput
              label="Tool Wear (min)"
              value={sensors.tool_wear}
              onChange={(value) =>
                setSensors({ ...sensors, tool_wear: value })
              }
            />
          </div>

          <button onClick={runPrediction} disabled={loading}>
            {loading ? "Analyzing..." : "Analyze Equipment"}
          </button>
        </section>

        <section className="analysis-grid">
          <div className="card">
            <h2>Anomaly Detection</h2>

            <div
              className={
                prediction?.anomaly_detected
                  ? "anomaly danger"
                  : "anomaly safe"
              }
            >
              <strong>
                {prediction
                  ? prediction.anomaly_detected
                    ? "ANOMALY DETECTED"
                    : "NORMAL"
                  : "WAITING"}
              </strong>

              <span>
                {prediction
                  ? `Score: ${prediction.anomaly_score.toFixed(3)}`
                  : "Run an analysis to check telemetry"}
              </span>
            </div>
          </div>

          <div className="card">
            <h2>Prediction History</h2>

            <ResponsiveContainer width="100%" height={250}>
              <LineChart data={history}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="id" />
                <YAxis domain={[0, 100]} />
                <Tooltip />
                <Line
                  type="monotone"
                  dataKey="risk"
                  strokeWidth={3}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </section>

        {error && <div className="error">{error}</div>}
      </main>
    </div>
  );
}

// Reusable sensor display card.
function SensorCard({ title, value }) {
  return (
    <div className="sensor-card">
      <span>{title}</span>
      <strong>{value}</strong>
    </div>
  );
}

// Reusable sensor input component.
function SensorInput({ label, value, onChange }) {
  return (
    <label className="input">
      <span>{label}</span>

      <input
        type="number"
        value={value}
        onChange={(event) => onChange(Number(event.target.value))}
      />
    </label>
  );
}

export default App;