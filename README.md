<div align="center">
    <a href="https://kubikk-ekkolodd.sintef.cloud/dashboard?id=SINDIT">
        <img src="https://kubikk-ekkolodd.sintef.cloud/api/project_badges/measure?project=SINDIT&metric=alert_status&token=sqb_daa44a05f36e549bc45f72c29dcb10b1b04bb781" alt="Quality Gate Status">
    </a>
    <a href="https://kubikk-ekkolodd.sintef.cloud/dashboard?id=SINDIT">
        <img src="https://kubikk-ekkolodd.sintef.cloud/api/project_badges/measure?project=SINDIT&metric=coverage&token=sqb_daa44a05f36e549bc45f72c29dcb10b1b04bb781" alt="Coverage">
    </a>
    <img src="https://img.shields.io/badge/code%20style-black-black" alt="Code Style Black">
    <img src="https://img.shields.io/badge/python-3.11-blue" alt="Python Version">
    <a href="https://pypi.org/project/sindit/">
        <img src="https://img.shields.io/pypi/v/sindit.svg" alt="PyPI version">
    </a>
</div>

<div align="center">
    <img src="https://raw.githubusercontent.com/SINTEF-9012/SINDIT20/refs/heads/main/src/sindit/docs/img/sindit_logo.png" alt="SINDIT Logo" width="350">
</div>

## Run backend using Docker Compose
To start the backend run (add the --build flag to build images before starting containers (build from scratch)):
```bash
docker-compose up
docker-compose up --build
```

This will build the GraphDB docker image and the FastAPI docker image.

The GraphDB instance will be available at: `localhost:7200`

The FastAPI documentation will be exposed at: `http://0.0.0.0:9017`

## Run backend locally
Desription of how to start the backend locally outside docker.
The backend consists of a GraphDB database and a FastAPI server.

### GraphDB
To start GraphDB, run these scripts from the GraphDB folder:
```bash
bash graphdb_install.sh
bash graphdb_preload.sh
bash graphdb_start.sh
```

To test your graphbd connection run from your base folder (/sindit):
```bash
python run_test.py
```

Go to localhost:7200 to configure graphdb

### API uvicorn server

First, ensure dependencies are installed:
```bash
uv pip install -e .
```
or
```bash
uv sync
```

To start the FastAPI server, run:
```bash
python run_sindit.py
```


### Run using vscode launcher

```bash
{
    "version": "0.2.0",
    "configurations": [

        {
            "name": "Python Debugger: Current File",
            "type": "debugpy",
            "request": "launch",
            "program": "${file}",
            "console": "integratedTerminal",
            "cwd": "${workspaceFolder}/src/sindit",
            "env": {
                "PYTHONPATH": "${workspaceFolder}/src"
            },
            "justMyCode": false
        }
    ]
}
```
## Using the API

### Authentication
The API requires a valid authentication token for most endpoints. Follow these steps to authenticate and use the API:

1. **Generate a Token**:
   - Use the `/token` endpoint to generate an access token.
   - Example `curl` command:
     ```bash
     curl -X POST "http://127.0.0.1:9017/token" \
     -H "Content-Type: application/x-www-form-urlencoded" \
     -d "username=new_user&password=new_password"
     ```
   - Replace `new_user` and `new_password` with the credentials provided below.

2. **Use the Token**:
   - Include the token in the `Authorization` header for all subsequent API calls:
     ```bash
     curl -X GET "http://127.0.0.1:9017/endpoint" \
     -H "Authorization: Bearer your_generated_token_here"
     ```

3. **Access API Documentation**:
   - The FastAPI documentation is available at: `http://127.0.0.1:9017/docs`

---

### Generate New Username and Password
To add a new user, update the `fake_users_db` in `authentication_endpoints.py` with the following credentials:

```python
fake_users_db = {
    "new_user": {
        "username": "new_user",
        "full_name": "New User",
        "email": "new_user@example.com",
        "hashed_password": "$2b$12$eW5j9GdY3.EciS3oKQxJjOyIpoUNiFZxrON4SXt3wVrgSbE1gDMba",  # Password: new_password
        "disabled": False,
    }
}
```

To generate a new hashed password, use the  Python snippet in `password_hash.py`.
Replace `"new_password"` with your desired password.

---

## AI Multi-Agent System (LLM / Chatbot)

This section documents the AI layer added on top of SINDIT for industrial CNC machine analysis. It is composed of a multi-agent architecture that answers natural-language questions by routing them to the right data source.

### Architecture overview

```
User question (natural language)
        │
        ▼
  IntegratedAgent  ←── semantic classification (embeddings + cosine similarity)
        │
        ├─── RetrieverAgent   →  Knowledge Graph (Text2SPARQL → GraphDB)
        ├─── AnalyticsAgent   →  Historical CNC data (Parquet files)
        └─── MonitoringAgent  →  Real-time MQTT data (SINTEF broker)
```

The `IntegratedAgent` classifies each question and delegates to one or several specialised agents:

| Agent | Data source | Use case |
|---|---|---|
| **RetrieverAgent** | GraphDB knowledge graph | Documentation, machine specs, component descriptions |
| **AnalyticsAgent** | Parquet files (`data/cnc/`) | Historical analysis of machined workpieces |
| **MonitoringAgent** | MQTT broker (real-time) | Live sensor values, current machine state |

### LLM — SSH tunnel to SINTEF mainframe

The LLM (Ollama) runs on the SINTEF mainframe. To expose it locally, open an SSH tunnel before starting the chatbot:

```bash
ssh -L 11434:localhost:11434 example@mainframe.sintef.no
```

This forwards `localhost:11434` on your machine to the Ollama server on the mainframe. The tunnel must stay open for the entire session.

### Launch the chatbot

```bash
streamlit run ui/src/chatbot/chatbot_ui.py --server.port 8502
```

Accessible at: `http://localhost:8502`

**Example questions:**

```
What is the description of the "Pneumatic System"?
What did I ask you in my previous question?
Analyse the workpiece OF10001 from 06:00 to 06:30
Show vibration for OF10005 from 2026-03-13 10:18 to 14:00
Analyse OF10001 from 06:00 to 08:00
```

### Launch the CNC visualization dashboard

Standalone Streamlit app for interactive exploration of workpiece data (time series, tool changes, temperatures, chatter statistics):

```bash
streamlit run sindit/src/sindit/processing/viz.py
```

### PDF → Knowledge Graph pipeline

Technical PDF documents (machine manuals, spare parts lists, etc.) can be ingested and stored in the knowledge graph. The LLM extracts entities (assets, properties, relationships) from each text chunk and stores them via the SINDIT REST API.

**Before running**, make sure the SSH tunnel is open (Ollama needed) and the SINDIT API is running (`localhost:9017`).

```bash
python -m sindit.src.sindit.knowledge_graph.pdf_file_to_kg
```

Configuration is done at the top of the `__main__` block in `pdf_file_to_kg.py`:

```python
PDF_PATH    = "data/documents/my_machine.pdf"
START_PAGE  = 5      # skip first N pages (cover, table of contents…)
START_CHUNK = 0      # resume from chunk N after a crash (0 = start fresh)
CLEAN_GRAPH = False  # True = wipe the KG before starting
GRAPH_NAME  = "default"
```

---

## CNC Data — Structure and Format

CNC data is stored under `data/cnc/` with one subfolder per work order (OF):

```
data/cnc/
├── OF10001/
│   ├── OF10001_G_BQC_S8CF2G_TYZBPS.csv   ← general machine state
│   ├── OF10001_G_BQC_S8CF2G_TYZBPS.parquet
│   ├── OF10001_G_BQC_S8CF2G_BXCZ3M.csv   ← axis power
│   ├── OF10001_G_BQC_S8CF2G_BXCZ3M.parquet
│   ├── OF10001_G_BQC_S8CF2G_7N4ZJ8.csv   ← vibration & chatter
│   └── OF10001_G_BQC_S8CF2G_7N4ZJ8.parquet
├── OF10002/ …
```

**Naming convention:** `OFxxxxx_MachineID_SensorType.csv`  
**Join key across files:** `timestamp` (ISO 8601 UTC)

### Available workpieces

| OF | Start | End | Duration |
|---|---|---|---|
| OF10001 | 2025-09-01 06:15 | 2025-09-02 00:22 | ~18 h |
| OF10002 | 2025-09-02 01:10 | 2025-09-02 16:17 | ~15 h |
| OF10003 | 2025-09-02 16:58 | 2025-09-03 08:41 | ~16 h |
| OF10004 | 2025-10-22 00:19 | 2025-10-22 13:23 | ~13 h |
| OF10005 | 2026-03-13 10:18 | 2026-03-16 00:54 | ~62 h |

### CSV → Parquet conversion

Raw CSV files must be converted to Parquet before the agents can use them (PyArrow predicate pushdown requires Parquet):

```bash
python -c "from sindit.src.sindit.processing.cnc_preprocessing import convert_all_cnc_data; convert_all_cnc_data()"
```

This iterates over all `data/cnc/OFxxxxx/*_TYZBPS.csv`, `*_BXCZ3M.csv`, and `*_7N4ZJ8.csv` files and writes the corresponding `.parquet` alongside each CSV.

### File types — columns and content

#### TYZBPS — General machine state (34 columns)

This is the main file. It captures the complete machine state at each timestamp: what it is doing, where it is, at what speed, with which tool, and how much energy it consumes.

| Column | Description |
|---|---|
| `timestamp` | Timestamp — join key across the 3 files |
| `Spindle_Speed_Actual` | Actual spindle speed (rpm) — > 0 means the machine is actively cutting |
| `Spindle_Speed_Commanded` | Spindle speed commanded by the NC program |
| `Spindle_Speed_Override` | Override applied by the operator (%) |
| `Feed_Rate_Actual` | Actual table feed rate (mm/min) — indicates a cutting move |
| `Feed_Rate_Commanded` | Feed rate commanded by the NC program |
| `Feed_Override` | Override applied by the operator (%) |
| `Position_MCS_X/Y/Z` | Absolute position in the Machine Coordinate System on the 3 linear axes |
| `Position_MCS_A/C` | Position of the rotary axes (5-axis machines only) |
| `Offset_X/Y/Z` | Axis offsets |
| `Power_Active` | Total active power consumed (W) |
| `Power_Apparent` | Apparent power (VA) |
| `Power_Reactive` | Reactive power (VAR) |
| `Power_Factor` | Power factor |
| `Power_Spindle` | Power consumed by the spindle motor alone (W) |
| `Energy_Total` | Cumulative energy counter (kWh) |
| `Tool_Number` | Active tool number (identifies which cutter is mounted) |
| `Tool_Length` | Active tool length (mm) |
| `Tool_Radius` | Active tool radius (mm) |
| `Program_Name` | Name of the NC program currently running |
| `Program_Block_Number` | Current NC block number (N…) |
| `Temperature_Head` | Head temperature (°C) |
| `Temperature_Room` | Room temperature (°C) |
| `Temperature_Y` | Y-axis temperature (°C) |
| `Temperature_Z` | Z-axis temperature (°C) |
| `Head_Angular_On` | Flag: angular head mode active |
| `Head_Auto_On` | Flag: automatic mode active |
| `Head_Boring_On` | Flag: boring mode active |
| `Operation_Mode` | Machine mode (automatic, manual, MDI…) |
| `Operation_Status` | Operation status (running, paused, emergency stop…) |

> **MCS** = Machine Coordinate System — the absolute reference frame fixed to the machine itself, as opposed to the workpiece coordinate system.  
> **Spindle** — the rotary motor that spins the cutting tool at high speed (1 000 – 20 000 rpm). When `Spindle_Speed_Actual > 0`, the machine is actively cutting.

#### BXCZ3M — Axis power and operation mode (7 columns)

Lighter file that provides per-axis power consumption and the machine's operational mode. Useful for identifying which axis is under load.

| Column | Description |
|---|---|
| `timestamp` | Timestamp — join key across the 3 files |
| `Power_X1` | Power consumed by motor 1 of the X axis (W) |
| `Power_X2` | Power consumed by motor 2 of the X axis (W) |
| `Power_Y` | Power consumed by the Y axis (W) |
| `Power_Z` | Power consumed by the Z axis (W) |
| `Operation_Mode` | Machine mode (automatic, manual, MDI…) |
| `Operation_Status` | Operation status (running, paused, emergency stop…) |

> **Two X motors:** the X axis uses a dual-drive configuration (one motor on each side of the gantry) to prevent racking over long travels. Each motor is monitored independently.

#### 7N4ZJ8 — Vibration and chatter detection (61 columns)

The most detailed file. It contains the full vibration signature of the machine: severity index, harmonic decomposition (FFT), spectral peaks, and a binary chatter detection flag.

| Column | Description |
|---|---|
| `timestamp` | Timestamp — join key across the 3 files |
| `Chatter_Detection_OnOff_X/Y` | Binary flag: chatter detected on axis X or Y |
| `Chatter_Detection_Amplitude_X/Y` | Amplitude of the detected chatter signal |
| `Chatter_Detection_Frequency_X/Y` | Frequency of the detected chatter (Hz) |
| `Vibration_Harmonic_N_X/Y_Amplitude` | Amplitude of the N-th harmonic (N = 1 to 8) |
| `Vibration_Harmonic_N_X/Y_Frequency` | Frequency of the N-th harmonic (Hz) |
| `Vibration_Peak_N_X/Y_Amplitude` | Amplitude of the N-th spectral peak (N = 1 to 5) |
| `Vibration_Peak_N_X/Y_Frequency` | Frequency of the N-th spectral peak (Hz) |
| `Vibration_Severity_X/Y` | Global vibration severity index (summary of all peaks) |

**Chatter** is a self-sustained resonant vibration between the tool and the workpiece. It degrades surface quality and accelerates tool wear. `Chatter_Detection_OnOff_X/Y = True` is an immediate alert.

**Harmonics (32 columns, Harmonic_1 to Harmonic_8):** the FFT of the vibration signal is decomposed into its fundamental frequency and its integer multiples. Harmonic_1 typically corresponds to the spindle rotation frequency; higher harmonics indicate more complex dynamics.

**Peaks (20 columns, Peak_1 to Peak_5):** the 5 frequencies with the highest amplitude in the full spectrum, regardless of their harmonic relationship. An unexpected peak at an unusual frequency may indicate a worn tool, loose fixture, or failing bearing.

**Vibration_Severity:** a single scalar summarising overall vibration intensity (computed following industrial norms such as ISO 10816). A low value means normal machining; a high value means excessive vibration → risk of poor surface finish or tool breakage.

---
