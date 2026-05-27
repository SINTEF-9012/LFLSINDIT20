"""
Monitoring Agent for Real-time CNC Machine Monitoring.

Flow:
  1. User asks a natural-language question about live machine data.
  2. The LLM classifies the query → returns which signals to subscribe to,
     how long to collect (period), and the sampling interval.
  3. The agent connects to the SINTEF MQTT broker, collects messages for
     `period` seconds, downsamples to one per `sample_interval`, then
     passes the samples to the LLM for a natural-language answer.
"""

import json
import os
import sys
import time
import logging
from random import randint
from typing import Any, Dict, List, Optional

from paho.mqtt import client as mqtt_client
from pydantic import BaseModel
from langchain_core.prompts import ChatPromptTemplate

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .llm_config import _get_llm


# ─────────────────────────────────────────────────────────────────────────────
# MQTT broker credentials (SINTEF CNC machines)
# ─────────────────────────────────────────────────────────────────────────────
MQTT_BROKER   = os.getenv("MQTT_BROKER",   "158.158.8.212")
MQTT_PORT     = int(os.getenv("MQTT_PORT", "1883"))
MQTT_USERNAME = os.getenv("MQTT_USERNAME", "reader_user")
MQTT_PASSWORD = os.getenv("MQTT_PASSWORD", "FpWWvYKMZ.ub3G!lAx6b3")

# Base topic — signals go after data/
# reed/machine/+/workorder/+/group/+/data/<signal>
MQTT_BASE_TOPIC = "reed/machine/+/workorder/+/group/+/data"

# ─────────────────────────────────────────────────────────────────────────────
# Known signals — these are the valid values that can appear after data/
# The LLM must return only values from this list.
# ─────────────────────────────────────────────────────────────────────────────
KNOWN_SIGNALS: List[str] = [
    # TYZBPS — general machine state
    "Spindle_Speed_Actual",
    "Spindle_Speed_Commanded",
    "Spindle_Speed_Override",
    "Feed_Rate_Actual",
    "Feed_Rate_Commanded",
    "Feed_Override",
    "Power_Active",
    "Power_Apparent",
    "Power_Reactive",
    "Power_Factor",
    "Power_Spindle",
    "Energy_Total",
    "Offset_X",
    "Offset_Y",
    "Offset_Z",
    "Position_MCS_X",
    "Position_MCS_Y",
    "Position_MCS_Z",
    "Position_MCS_A",
    "Position_MCS_C",
    "Temperature_Head",
    "Temperature_Room",
    "Temperature_Y",
    "Temperature_Z",
    "Tool_Number",
    "Tool_Length",
    "Tool_Radius",
    "Program_Name",
    "Program_Block_Number",
    "Head_Angular_On",
    "Head_Auto_On",
    "Head_Boring_On",
    "Operation_Mode",
    "Operation_Status",
    # BXCZ3M — axis power
    "Power_X1",
    "Power_X2",
    "Power_Y",
    "Power_Z",
    # 7N4ZJ8 — vibration and chatter
    "Chatter_Detection_OnOff_X",
    "Chatter_Detection_OnOff_Y",
    "Chatter_Detection_Amplitude_X",
    "Chatter_Detection_Amplitude_Y",
    "Chatter_Detection_Frequency_X",
    "Chatter_Detection_Frequency_Y",
    "Vibration_Severity_X",
    "Vibration_Severity_Y",
    "Vibration_Harmonic_1_X_Amplitude",
    "Vibration_Harmonic_1_Y_Amplitude",
    "Vibration_Peak_1_X_Amplitude",
    "Vibration_Peak_1_Y_Amplitude",
]


class MQTTQueryParams(BaseModel):
    """
    Parameters extracted by the LLM from the user's query.

    signals         : list of signal names (must be from KNOWN_SIGNALS)
                      that are relevant to the user's question.
                      These will be appended after data/ in the MQTT topic.
    period          : total duration (seconds) to listen on the broker.
    sample_interval : one message is kept per this many seconds (downsampling).
    """
    signals: List[str]
    period: float
    sample_interval: float


def validate_mqtt_params(params: MQTTQueryParams) -> MQTTQueryParams:
    """
    Validate that the LLM returned sensible MQTT query parameters.

    - Filters out signal names not in KNOWN_SIGNALS.
    - Falls back to a safe default if signals list is empty after filtering.
    - Clamps period and sample_interval to reasonable bounds.

    Returns a corrected MQTTQueryParams.
    """
    # Filter signals to only known ones
    valid_signals = [s for s in params.signals if s in KNOWN_SIGNALS]

    if not valid_signals:
        logging.warning(
            "[MonitoringAgent] LLM returned no valid signals %s — "
            "falling back to Spindle_Speed_Actual + Power_Active",
            params.signals,
        )
        valid_signals = ["Spindle_Speed_Actual", "Power_Active"]

    unknown = [s for s in params.signals if s not in KNOWN_SIGNALS]
    if unknown:
        logging.warning("[MonitoringAgent] Unknown signals ignored: %s", unknown)

    # Clamp period: between 2s and 60s
    period = max(2.0, min(float(params.period), 60.0))

    # Clamp sample_interval: between 0.5s and period
    sample_interval = max(0.5, min(float(params.sample_interval), period))

    return MQTTQueryParams(
        signals=valid_signals,
        period=period,
        sample_interval=sample_interval,
    )


# ─────────────────────────────────────────────────────────────────────────────
# MQTT tool
# ─────────────────────────────────────────────────────────────────────────────

_CLASSIFIER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a signal classifier for CNC machine MQTT data.

The MQTT topic structure is:
  reed/machine/+/workorder/+/group/+/data/<signal_name>

The available signal names are:
{known_signals}

Given the user's question, return a JSON object with exactly these fields:
{{
  "signals": ["Signal_Name_1", "Signal_Name_2"],
  "period": <float, seconds to listen, between 5 and 30>,
  "sample_interval": <float, seconds between samples, between 1 and 5>
}}

Rules:
- Only use signal names from the list above.
- Choose signals that are directly relevant to the question.
- If the question is general, return the most common signals: Spindle_Speed_Actual, Feed_Rate_Actual, Power_Active.
- Respond ONLY with the JSON object, no explanation.
"""),
    ("human", "{query}"),
])

_ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a real-time monitoring agent for CNC industrial machines.
You receive MQTT messages collected live from the machines and must answer
the user's question based on this data.

Each sample contains:
- topic        : full MQTT topic (encodes machine id, workorder, group, signal)
- payload      : the sensor value at that moment
- received_at  : Unix timestamp

When answering:
1. Extract the relevant signal values from the samples.
2. Summarise min / max / average where relevant.
3. Flag any anomalies (e.g. chatter detected, abnormal temperature).
4. Be concise and technical.
5. If no data was received, say so clearly.
"""),
    ("human", "User question: {question}\n\nMQTT samples:\n{samples}"),
])


class SINDITMQTTTool:
    """
    Connects to the SINTEF MQTT broker, collects messages for a given period,
    and downsamples to one message per sample_interval window.
    """

    def connect_mqtt(self) -> mqtt_client.Client:
        """Create and connect a paho MQTT client to the SINTEF broker."""
        client_id = f"sindit-monitor-{randint(0, 9999)}"

        def on_connect(client, userdata, flags, rc):
            if rc == 0:
                logging.info("[MQTT] Connected to broker")
            else:
                logging.error(f"[MQTT] Connection failed (rc={rc})")

        def on_disconnect(client, userdata, rc):
            logging.info(f"[MQTT] Disconnected (rc={rc})")

        c = mqtt_client.Client(client_id=client_id)
        c.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)
        c.on_connect = on_connect
        c.on_disconnect = on_disconnect
        c.connect(MQTT_BROKER, MQTT_PORT, keepalive=60)
        return c

    def retrieve_data(
        self,
        signals: List[str],
        period: float,
        sample_interval: float,
    ) -> List[Dict]:
        """
        Subscribe to one topic per signal, collect for `period` seconds,
        then downsample to one message per `sample_interval` window.

        Args:
            signals         : list of signal names (appended after data/)
            period          : total listening time in seconds
            sample_interval : one sample kept per this many seconds

        Returns:
            List of dicts {topic, payload, received_at}
        """
        messages = []

        def on_message(client, userdata, msg):
            try:
                payload = json.loads(msg.payload.decode())
            except Exception:
                payload = msg.payload.decode()
            messages.append({
                "topic":       msg.topic,
                "payload":     payload,
                "received_at": time.time(),
            })

        client = self.connect_mqtt()
        client.on_message = on_message

        # Subscribe to one topic per signal
        for signal in signals:
            full_topic = f"{MQTT_BASE_TOPIC}/{signal}"
            client.subscribe(full_topic)
            logging.info(f"[MQTT] Subscribed to {full_topic}")

        logging.info(f"[MQTT] Listening for {period}s (sample_interval={sample_interval}s)...")
        client.loop_start()
        time.sleep(period)
        client.loop_stop()
        client.disconnect()
        logging.info(f"[MQTT] Collected {len(messages)} raw message(s)")

        if not messages:
            return []

        #keep the last message per sample_interval window
        t0 = messages[0]["received_at"]
        windows: Dict[int, Dict] = {}
        for msg in messages:
            window_index = int((msg["received_at"] - t0) / sample_interval)
            windows[window_index] = msg  # overwrite → keeps latest in each window

        samples = [windows[k] for k in sorted(windows)]
        logging.info(f"[MQTT] Downsampled to {len(samples)} sample(s)")
        return samples


# ─────────────────────────────────────────────────────────────────────────────
# Monitoring Agent
# ─────────────────────────────────────────────────────────────────────────────

class MonitoringAgent:
    """
    Monitoring Agent for real-time CNC machine data via MQTT.

    Step 1 — classify: LLM extracts signals, period, sample_interval from query.
    Step 2 — validate: check LLM output against KNOWN_SIGNALS list.
    Step 3 — collect:  connect to broker, gather samples.
    Step 4 — answer:   LLM formulates a natural-language answer from the samples.
    """

    def __init__(self) -> None:
        self._llm  = _get_llm()
        self._tool = SINDITMQTTTool()

    def classify_query(self, user_query: str) -> MQTTQueryParams:
        """
        Ask the LLM which signals to subscribe to, and for how long.
        Validates and sanitises the response before returning.
        """
        chain = _CLASSIFIER_PROMPT | self._llm.with_structured_output(MQTTQueryParams)
        raw: MQTTQueryParams = chain.invoke({
            "known_signals": "\n".join(f"  - {s}" for s in KNOWN_SIGNALS),
            "query": user_query,
        })
        logging.info(f"[MonitoringAgent] LLM classified → signals={raw.signals}, "
              f"period={raw.period}s, sample_interval={raw.sample_interval}s")
        validated = validate_mqtt_params(raw)
        logging.info(f"[MonitoringAgent] After validation → signals={validated.signals}, "
              f"period={validated.period}s, sample_interval={validated.sample_interval}s")
        return validated

    def get_realtime_context(self, query: str) -> Dict[str, Any]:
        """Classify query → collect MQTT data → return raw context dict."""
        params = self.classify_query(query)
        samples = self._tool.retrieve_data(
            signals=params.signals,
            period=params.period,
            sample_interval=params.sample_interval,
        )
        return {
            "samples": samples,
            "signals_used": params.signals,
            "period": params.period,
            "sample_interval": params.sample_interval,
        }

    def query(self, user_query: str) -> str:
        """
        Full pipeline: classify → collect → answer.

        Args:
            user_query: Natural-language question about live machine data.

        Returns:
            str: LLM answer grounded in the collected MQTT samples.
        """
        context = self.get_realtime_context(user_query)
        samples = context["samples"]

        if not samples:
            return (
                "No MQTT data was received during the collection window. "
                "The broker may be unreachable (check VPN / network) or no machine "
                "is currently publishing on the subscribed topics."
            )

        chain = _ANSWER_PROMPT | self._llm
        result = chain.invoke({
            "question": user_query,
            "samples":  json.dumps(samples, indent=2, default=str),
        })
        return result.content if hasattr(result, "content") else str(result)
