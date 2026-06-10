import re
import json
import logging
import os
import time

import requests
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from .llm_config import _get_llm

load_dotenv()

DPP_API_URL = os.environ.get("DPP_API_URL", "http://localhost:8888")
USE_MOCK = os.environ.get("USE_MOCK", "True") == "True"
DEFAULT_BATCH = "OF10005"

QA_TEMPLATE = """You are a Green Agent specialized in carbon footprint analysis
for CNC manufacturing operations. Answer the question using the DPP data below.

Lifecycle phases (ISO 14044):
- A1: Raw material extraction
- A2: Transport to factory
- A3: Manufacturing (CNC machine)
- A1-A3: Total cradle-to-gate

DPP Data:
{context}

Question: {question}

Answer in the same language as the question. Be precise with units (kg CO2eq).
"""

QA_PROMPT = PromptTemplate(input_variables=["question", "context"], template=QA_TEMPLATE)


class GreenAgent:

    def __init__(self):
        self._llm = _get_llm()

    def _extract_batch_id(self, question: str) -> str:
        match = re.search(r'OF\d+', question, re.IGNORECASE)
        return match.group(0).upper() if match else DEFAULT_BATCH

    def _get_data(self, batch_id: str) -> dict:
        print(f"USE_MOCK : {USE_MOCK}")
        if USE_MOCK:
            path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "data", "cnc", batch_id, f"DPP_goimek_{batch_id}_G_BQC_S8CF2G_2026-06-01.json")
            with open(path) as f:
                print(f"path loaded: {path}")
                return json.load(f)

        response = requests.get(f"{DPP_API_URL}/batch/{batch_id}", timeout=10)
        response.raise_for_status()
        return response.json()

    def _summarize_dpp(self, data: dict) -> str:
        entry = data["DPP"][0]
        footprints = entry["carbonFootprint"]["ProductCarbonFootprints"]

        lines = [
            f"Company : {entry['company']}",
            f"Event : {entry['event_id']}",
            f"Product : {entry['product']}",
            ""
        ]
        for e in footprints:
            phase = e["LifeCyclePhases"][0]["LifeCyclePhase"]
            value = e["PCF in kg CO2eq"]
            lines.append(f"Phase {phase}: {value} kg CO2eq")

        return "\n".join(lines)

    def _run(self, question: str) -> str:
        try:
            batch_id = self._extract_batch_id(question)
            data = self._get_data(batch_id)
            context = self._summarize_dpp(data)
            prompt = QA_PROMPT.format(question=question, context=context)
            response = self._llm.invoke(prompt)
            return response.content if hasattr(response, "content") else str(response)
        except Exception as e:
            logging.error(f"[GreenAgent] {e}")
            return f"Error retrieving DPP data: {e}"