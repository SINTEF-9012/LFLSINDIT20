# -*- coding: utf-8 -*-
"""Graph Transformer Utilities.
This module provides utilities for transforming and extracting knowledge graphs
import logging
from documents. It includes functions to create extraction chains,
map nodes and relationships to base types, and store the extracted graphs
in a Neo4j database.
"""
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from typing import Optional, List
from pydantic import ValidationError
from langdetect import detect
from ..llm.llm_config import _get_llm
from ..util.sindit_client import SINDITClient #class to store graph
from  .graph_model_for_llm import SINDITKnowledgeGraph, SINDITProperty #Graph
from ..util.sindit_client import SINDITClient
import re
import logging
def detect_language(text: str) -> str:
    """
    Detect the language of the input text.
    
    Args:
        text (str): Input text to analyze
        
    Returns:
        str: Detected language code (e.g., 'en', 'es', 'de')
    """
    try:
        # Take a sample of text for detection (first 500 characters should be enough)
        sample_text = text[:500].strip()
        if len(sample_text) < 10:
            return 'en'  # Default to English for very short texts
        
        detected = detect(sample_text)
        return detected
    except Exception:
        # Default to English if detection fails
        return 'en'

def get_language_name(lang_code: str) -> str:
    """Convert language code to language name."""
    language_map = {
        'en': 'English',
        'es': 'Spanish', 
        'de': 'German',
        'fr': 'French',
        'it': 'Italian',
        'pt': 'Portuguese'
    }
    return language_map.get(lang_code, 'English')

def sanitize_property_value(value: str) -> str:
    """Clears a property value for the SINDIT API."""
    if not value:
        return ""
    # Remove control characters and non-printable characters
    value = re.sub(r'[\x00-\x1f\x7f-\x9f]', ' ', value)
    value = ' '.join(value.split())
    return value[:500]

def get_extraction_chain(
    allowed_nodes: Optional[List[str]] = None,
    allowed_rels: Optional[List[str]] = None,
    source_language: Optional[str] = None
    ):

    # Language-specific instructions
    if source_language and source_language.lower() != "english":
        language_instruction = f"""
## Language Instructions
- The input text is in {source_language}. Extract ALL content in English.
- Translate asset names, labels and descriptions to English.
- Asset IDs must use English terms (e.g., "Welding_Machine", "Control_System").
- Example translations:
  - Spanish "Máquina de Soldadura" → id: "Welding_Machine", label: "Welding Machine"
  - Spanish "Sistema de Control"   → id: "Control_System",  label: "Control System"
  - Spanish "Bomba"                → id: "Pump_P1",         label: "Pump P1"
"""
    else:
        language_instruction = """
## Language Instructions
- The input text is in English. Use clear English terms for all ids and labels.
"""

    # Build the system prompt aligned with SINDITKnowledgeGraph schema
    system_prompt = """You are an expert algorithm for extracting industrial knowledge graphs from technical documentation.
Your output MUST strictly follow the SINDIT schema described below.

""" + language_instruction + """
## SINDIT Schema

### assets  (List of SINDITAsset)
Each asset represents an industrial component (machine, sub-system, sensor, valve, motor, controller...).
Each asset MUST have:
- "id"              : unique English identifier, no spaces, use underscores (e.g. "Pump_P401", "Motor_M1", "PLC_Siemens")
- "label"           : human-readable name in English (e.g. "Pump P-401", "Main Motor", "Siemens PLC")
- "assetType"       : category of the component. Use one of:
                      Pump, Motor, Valve, Sensor, Controller, Compressor, HeatExchanger,
                      Tank, Conveyor, Robot, Drive, PLC, HMI, Filter, Fan, Actuator, System, Subsystem
                      — or any other precise English technical term if none fits.
- "assetDescription": one sentence describing what this component does (in English).
- "properties"      : list of technical parameters and specifications for THIS asset only.
                      Each property MUST have:
                        - "propertyName"       : camelCase key (e.g. "maxPressure", "ratedPower", "rotationSpeed")
                        - "propertyValue"      : value as a string (e.g. "250", "10.5", "enabled")
                        - "propertyUnit"       : unit of measurement or null (e.g. "bar", "kW", "rpm", "V", "Hz")
                        - "propertyDescription": short explanation of the parameter, or null

### relationships  (List of SINDITRelationship)
Each relationship connects two assets.
Each relationship MUST have:
- "sourceId"              : the "id" of the source asset (must match an existing asset id)
- "targetId"              : the "id" of the target asset (must match an existing asset id)
- "relationshipType"      : EXACTLY one of these values (case-sensitive, camelCase):
                            consistsOf, partOf, connectedTo, dependsOn, derivedFrom,
                            monitors, controls, simulates, uses, communicatesWith, isTypeOf
- "relationshipDescription": one sentence explaining the relationship (in English), or null

## Rules
1. Every asset ID must be unique and contain no spaces.
2. Every relationshipType must be one of the 11 allowed values above — never invent new types.
3. sourceId and targetId must refer to assets that exist in your "assets" list.
4. Do NOT create a separate asset for a pure numeric value or date — add it as a property.
5. Merge duplicate mentions of the same component into a single asset with the most complete information.
6. Extract ALL identifiable components, not just the main machine.
7. If a technical parameter has a unit (bar, kW, rpm...), always fill "propertyUnit".
"""

    if allowed_nodes:
        system_prompt += f"\n## Allowed assetType values (restrict to these): {', '.join(allowed_nodes)}\n"
    if allowed_rels:
        system_prompt += f"## Allowed relationshipType values (restrict to these): {', '.join(allowed_rels)}\n"

    system_prompt += """
## Output example (partial)
{{
  "assets": [
    {{
      "id": "Pump_P401",
      "label": "Pump P-401",
      "assetType": "Pump",
      "assetDescription": "Centrifugal pump responsible for fluid circulation in the cooling circuit.",
      "properties": [
        {{"propertyName": "maxPressure",    "propertyValue": "10",   "propertyUnit": "bar", "propertyDescription": "Maximum operating pressure"}},
        {{"propertyName": "ratedPower",     "propertyValue": "7.5",  "propertyUnit": "kW",  "propertyDescription": "Nominal motor power"}},
        {{"propertyName": "rotationSpeed",  "propertyValue": "1450", "propertyUnit": "rpm", "propertyDescription": "Nominal rotation speed"}}
      ]
    }},
    {{
      "id": "Motor_M1",
      "label": "Motor M1",
      "assetType": "Motor",
      "assetDescription": "Electric motor driving pump P-401.",
      "properties": [
        {{"propertyName": "voltage",    "propertyValue": "400", "propertyUnit": "V",  "propertyDescription": "Supply voltage"}},
        {{"propertyName": "frequency",  "propertyValue": "50",  "propertyUnit": "Hz", "propertyDescription": "Supply frequency"}}
      ]
    }}
  ],
  "relationships": [
    {{
      "sourceId": "Motor_M1",
      "targetId": "Pump_P401",
      "relationshipType": "controls",
      "relationshipDescription": "Motor M1 drives pump P-401 via a direct shaft coupling."
    }}
  ]
}}

## Strict Compliance
Respond ONLY with the structured output. Do not add explanations outside the schema."""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Extract the SINDIT knowledge graph from the following industrial documentation:\n\n{input}"),
    ])
    llm = _get_llm()

    # ── DEBUG: log the raw LLM response before parsing ─────────────────────────
    import os
    if os.getenv("DEBUG_LLM", "0") == "1":
        from langchain_core.runnables import RunnableLambda
        def _log_raw(x):
            logging.info("\n" + "="*60)
            logging.info("🔍 RAW LLM OUTPUT:")
            logging.info(x.content if hasattr(x, "content") else str(x))
            logging.info("="*60 + "\n")
            return x
        raw_chain = prompt | llm | RunnableLambda(_log_raw)
        # Runs extraction in raw mode to inspect output, then continues normally
    # ───────────────────────────────────────────────────────────────────────────

    structured_llm = llm.with_structured_output(SINDITKnowledgeGraph)
    return prompt | structured_llm


def extract_graph_only(
    document: Document,
    allowed_asset_types: Optional[List[str]] = None,
    allowed_relationship_types: Optional[List[str]] = None,
    source_label: Optional[str] = None,
) -> "SINDITKnowledgeGraph":
    """Extract a SINDITKnowledgeGraph from a document chunk.

    Args:
        document: LangChain chunk (page_content + metadata).
        allowed_asset_types: Whitelist of asset types (e.g. ["Pump", "Motor"]).
                             None = all types allowed.
        allowed_relationship_types: Whitelist of relationship types
                                    (e.g. ["controls", "partOf"]).
                                    None = all types allowed.
        source_label: Source label for tracing (not used during extraction).
    """
    # Detect language automatically using langdetect
    lang_code = detect_language(document.page_content)
    detected_language = get_language_name(lang_code)

    # Extract graph data using ChatOllama functions with language awareness
    extract_chain = get_extraction_chain(allowed_asset_types, allowed_relationship_types, detected_language)
    
    import time as _time

    max_retries = 3
    page_val = document.metadata.get("page", "?")
    source_val = document.metadata.get("source", "")

    for attempt in range(max_retries):
        try:
            print(f"[LLM] Attempt {attempt+1}/{max_retries} — page {page_val}, {len(document.page_content)} chars — calling LLM...")
            _t0 = _time.time()
            # Pass the content with the correct key name 'input'
            Knowledge_graph = extract_chain.invoke({"input": document.page_content})
            elapsed = _time.time() - _t0
            logging.info(f"[LLM] Done in {elapsed:.1f}s — {len(Knowledge_graph.assets)} assets, {len(Knowledge_graph.relationships)} relationships")
            source_val = document.metadata.get("source", "")
            page_val = document.metadata.get("page", "")
            page_content = document.page_content or "" 
            for asset in Knowledge_graph.assets:
                if asset.properties is None:
                    asset.properties = []
                    
                asset.properties.append(SINDITProperty(propertyName="source", propertyValue=source_val, propertyUnit=None, propertyDescription="Source file of this asset"))
                asset.properties.append(SINDITProperty(propertyName="page", propertyValue=str(page_val), propertyUnit=None, propertyDescription="Page number in the source document"))
                # page_content truncated to 500 chars to avoid oversized payloads
                asset.properties.append(SINDITProperty(propertyName="page_content", propertyValue=sanitize_property_value(page_content), propertyUnit=None, propertyDescription="Excerpt from the source chunk"))
            
            # for rel in Knowledge_graph.relationships:
            #     if rel.properties is None:
            #         rel.properties = []
            #     rel.properties.append(Property(key="source", value=source_val))
            #     rel.properties.append(Property(key="page", value=str(page_val)))
            #     rel.properties.append(Property(key="page_content", value=page_content))
            
            
            # Ensure rels field exists and is a list
            if not hasattr(Knowledge_graph, 'relationships') or Knowledge_graph.relationships is None:
                Knowledge_graph.relationships = []
            
            # Ensure nodes field exists and is a list  
            if not hasattr(Knowledge_graph, 'assets') or Knowledge_graph.assets is None:
                Knowledge_graph.assets = []
            
            # If we get here, the extraction was successful
            break
                
        except ValidationError as e:
            if attempt < max_retries - 1:
                logging.error(f"[RETRY {attempt+1}/3] Chunk page={page_val} — ValidationError: {e}")
                continue
            else:
                logging.error(f"ValidationError after {max_retries} attempts: {e}. Returning empty graph... Make sure your Ollama port is accessible")
                # exit(1)
                return SINDITKnowledgeGraph(
                    assets=[],
                    relationships=[],
                )
                
        except Exception as e:
            logging.error(f"Unexpected error during extraction: {e}. Returning empty graph... Make sure your Ollama port is accessible")
            exit(1)
            # return SINDITKnowledgeGraph(
            #     assets=[],
            #     relationships=[],
            # )

    return Knowledge_graph

def extract_and_store_graph(
    document: Document,
    allowed_asset_types: Optional[List[str]] = None,
    allowed_relationship_types: Optional[List[str]] = None,
    source_label: Optional[str] = None,
) -> "SINDITKnowledgeGraph":
    """Extract the graph from a chunk AND store it in SINDIT via REST API."""
    knowledge_graph = extract_graph_only(
        document,
        allowed_asset_types=allowed_asset_types,
        allowed_relationship_types=allowed_relationship_types,
        source_label=source_label,
    )
    sindit_client = SINDITClient()
    sindit_client.store_graph(knowledge_graph)
    return knowledge_graph