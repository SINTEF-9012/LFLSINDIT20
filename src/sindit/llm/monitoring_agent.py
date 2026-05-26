"""
Monitoring Agent for Real-time Manufacturing Process Monitoring.

This module provides real-time monitoring capabilities for manufacturing processes,
including anomaly detection, status tracking, and proactive alerting for
manufacturing operations.
"""

import json
import os
import sys
from typing import Any, Dict, Optional

# Add the src directory to the path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.prompts import (ChatPromptTemplate, HumanMessagePromptTemplate,
                               SystemMessagePromptTemplate)
from langchain.tools import BaseTool
from pydantic import BaseModel

from .llm_config import _get_llm
from ..util.sindit_client import SINDITClient

class MonitoringQuery(BaseModel):
    query: str
    asset_type: Optional[str] = None
    property_type: Optional[str] = None

client = SINDITClient()

class SINDITDataTool(BaseTool):
    name: str = "sindit_data_tool"
    description: str = "Tool for retrieving real-time manufacturing data from SINDIT Knowledge Graph API."
    args_schema: type[MonitoringQuery] = MonitoringQuery

    def classify_query(self, query: str) -> Dict[str, Any]:
        key_assets = [
            "http://sindit.sintef.no/2.0#factory-sensor",
            "http://sindit.sintef.no/2.0#camera",
            "http://sindit.sintef.no/2.0#vgr",
            "http://sindit.sintef.no/2.0#hbw",
            "http://sindit.sintef.no/2.0#sld",
            "http://sindit.sintef.no/2.0#mpo",
            "http://sindit.sintef.no/2.0#dso",
            "http://sindit.sintef.no/2.0#dsi",
            "http://sindit.sintef.no/2.0#order",
        ]

        asset_keywords = {
            "vgr": ["vgr", "vacuum gripper", "gripper", "robot"],
            "hbw": ["hbw", "high bay warehouse", "warehouse", "storage"],
            "sld": ["sld", "sorting line", "detection", "sorting"],
            "mpo": ["mpo", "multi processing", "oven", "processing station"],
            "dso": ["dso", "outgoing", "output", "piece out"],
            "dsi": ["dsi", "incoming", "input", "piece in"],
            "factory-sensor": ["temperature", "humidity", "air quality", "brightness", "sensor", "environmental", "environment"],
            "camera": ["camera", "vision", "image", "picture"],
            "order": ["order", "production order", "manufacturing order", "order status"],
            "Stock": ["stock", "inventory", "material"]
        }

        query_lower = query.lower()

        # Find specific assets mentioned in the query
        relevant_assets = []
        for asset_key, keywords in asset_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                asset_uri = f"http://sindit.sintef.no/2.0#{asset_key}"
                if asset_uri in key_assets:
                    relevant_assets.append(asset_uri)

        # If no specific assets found, default to key manufacturing assets
        if not relevant_assets:
            return key_assets

        return relevant_assets

    def _run(self, query: str, **kwargs):
        """
        Retrieve real-time data from SINDIT DT Platform based on the query.

        Args:
            query: Natural language query about manufacturing data

        Returns:
            dict: Real-time data from SINDIT DT Platform
        """

        relevant_assets = self.classify_query(query)

        # Get available data using correct endpoints
        streaming_data = {}

        # Get all node types to understand available data structure
        node_types = client.p('kg/node_types')
        streaming_data['node_types'] = node_types

        assets = []

        for asset_uri in relevant_assets:
            asset_data = client.query_get_api('kg/node', asset_uri, 'node_uri')
            assets.append(asset_data)

        streaming_data['data'] = assets

        # Get connection information
        connection_data = client.query_get_api('kg/node', "http://sindit.sintef.no/2.0#mqtt-connection", 'node_uri')
        streaming_data['connections'] = [connection_data] if connection_data else []

        return streaming_data

class MonitoringAgent:
    """
    Monitoring Agent for real-time manufacturing data retrieval and analysis.

    This agent connects to the SINDIT Knowledge Graph API to retrieve real-time
    manufacturing data and provides insights about factory operations, sensor status,
    and equipment monitoring.
    """

    def __init__(self):
        """Initialize the Monitoring Agent with necessary tools and LLM."""

        # Initialize tools
        self.sindit_data_tool = SINDITDataTool()
        # self.status_analyzer = ManufacturingStatusAnalyzer()

        # Initialize chat model with flexible LLM config
        self._llm = _get_llm()

        # Initialize conversation chain
        self.initialize_conversation_chain()

    def initialize_conversation_chain(self):
        """Initialize the conversation chain for the monitoring agent."""

        self.system_msg_template = SystemMessagePromptTemplate.from_template(
            template="""
            You are a specialized Monitoring Agent for the fischertechnik Training Factory Industry 4.0 system.
            Your expertise lies in real-time data monitoring, sensor analysis, and manufacturing status assessment.

            Your capabilities include:
            - Retrieving real-time sensor data from the SINDIT Knowledge Graph
            - Analyzing manufacturing equipment status (VGR, HBW, SLD, MPO, DSO, DSI)
                - Manufacturing equipment status (VGR, HBW, SLD, MPO, DSO, DSI)
                - VGR: Vacuum Gripper Robot
                - HBW: High Bay Warehouse
                - SLD: Sorting Line with Detection
                - MPO: Multi Processing Station with Oven
                - DSI/DSO: Sensor for incoming/outgoing pieces
            - Monitoring environmental conditions (temperature, humidity, air quality, brightness)
            - Assessing data connection health and streaming properties
            - Providing insights on factory operational status

            When answering questions about manufacturing status or sensor data:
            1. Always retrieve the latest real-time data first
            2. Analyze the data for patterns, anomalies, or status indicators
            3. Provide related information about factory operations
            4. Reference specific sensor readings and asset statuses when available

            Be technical but accessible, focusing on operational insights that help with:
            - Production monitoring and control
            - Preventive maintenance planning
            - Environmental condition assessment
            - System health and connectivity status
            - Performance optimization opportunities

            If real-time data is unavailable or incomplete, clearly indicate what information is missing
            and suggest alternative monitoring approaches.
            """
        )

        self.human_msg_template = HumanMessagePromptTemplate.from_template(template="{input}")

        self.prompt_template = ChatPromptTemplate.from_messages([
            self.system_msg_template,
            self.human_msg_template
        ])

    def get_realtime_context(self, query: str) -> Dict[str, Any]:
        """
        Get real-time monitoring context.

        Args:
            query: User's question

        Returns:
            dict: Real-time monitoring context
        """

        # Get real-time data
        realtime_data = self.sindit_data_tool.run(query)

        return {
            "realtime_data": realtime_data,
            "source": "MonitoringAgent"
        }

    def query(self, user_query: str) -> str:
        """
        Process a user query about manufacturing monitoring.

        Args:
            user_query: User's question about manufacturing status or real-time data

        Returns:
            str: Comprehensive response with real-time data analysis
        """
        # Retrieve real-time data relevant to the query by using the SINDIT data tool
        real_time_context = self.get_realtime_context(user_query)
        real_time_data = real_time_context.get("realtime_data", {})

        # Format the input for the conversation chain
        formatted_input = f"""
        User Query: {user_query}

        Real-time Data Retrieved:
        {json.dumps(real_time_data, indent=2)}

        Please provide a comprehensive response based on this real-time manufacturing data.
        """

        # Generate response using the conversation chain
        response_result = self._llm.invoke({"input": formatted_input})
        
        # Extract the text content from the response
        if isinstance(response_result, dict):
            response = response_result.get('text', str(response_result))
        else:
            response = str(response_result)

        return response

    def get_sindit_status(self) -> Dict[str, Any]:
        """
        Retrieve connection info of MQTT connections from SINDIT Knowledge Graph API.

        Returns:
            dict: Real-time data from SINDIT DT Platform
        """

        return client.query_get_connections_info('kg/node', "http://sindit.sintef.no/2.0#mqtt-connection")