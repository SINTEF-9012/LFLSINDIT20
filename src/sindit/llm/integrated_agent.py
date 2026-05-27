"""
Integrated Manufacturing Agent for Comprehensive Manufacturing Intelligence.

This module provides an integrated agent that combines retrieval, monitoring,
and analytics capabilities to provide comprehensive manufacturing intelligence
and decision support.
"""

import logging
import json
import os
import numpy as np
import re
from datetime import datetime
from typing import Any, Dict, Optional, Tuple, List

from langchain_core.prompts import (ChatPromptTemplate, HumanMessagePromptTemplate,
                                    SystemMessagePromptTemplate)
from .llm_config import _get_llm

from .monitoring_agent import MonitoringAgent
from .retriver_agent import RetrieverAgent
from .analytics_agent import AnalyticsAgent
from ..util.embedding import embedding_model

class IntegratedAgent:
    """
    Integrated Manufacturing Agent that combines retriever, monitoring, and analytics capabilities.

    This agent provides comprehensive manufacturing support by:
    1. Retrieving documentation and manuals from GraphDB (via RetrieverAgent)
    2. Monitoring real-time manufacturing data from SINDIT DT platform (via MonitoringAgent)
    3. Analyzing historical manufacturing data with charts and insights (via AnalyticsAgent)
    4. Providing unified responses that combine static knowledge, live data, and historical analytics
    """

    def __init__(self):
        """Initialize the Integrated Agent with retriever, monitoring, and analytics capabilities."""

        # MonitoringAgent is optional — set to None if not available
        self.monitoring_agent = None
        # self.monitoring_agent = MonitoringAgent()

        self.retriever_agent = RetrieverAgent()
        self.analytics_agent = AnalyticsAgent()

        self.llm = _get_llm()
        self.embeddings = embedding_model
        self.last_parsed_time_range = None
        # Last Plotly figure produced by the analytics agent — read by chatbot_ui to display the chart
        self.last_figure = None

        # Initialize semantic classification system
        self._initialize_semantic_classification()
        self.initialize_conversation_chain()

    def _initialize_semantic_classification(self):
        """Initialize the semantic classification system with example questions and embeddings."""
        
        # Curated taxonomy of CNC manufacturing use cases with example questions
        self.example_questions = {
            'documentation': [
                # CNC machine documentation and manuals
                "What are the specifications of the CNC milling machine?",
                "How to perform tool change on the CNC machine?",
                "What is the procedure for CNC machine maintenance?",
                "How to calibrate the spindle on the milling machine?",
                "What are the safety procedures for CNC operation?",
                "How to set cutting parameters for different materials?",
                "What is the maximum spindle speed of the machine?",
                "How to program a G-code for a new workpiece?",
                "What are the recommended feed rates for aluminum?",
                "How to set up workpiece coordinate systems?",
                "What are the troubleshooting steps for spindle overload?",
                "How to replace a worn cutting tool?",
                "What are the lubrication requirements for the CNC machine?",
                "How to configure axis limits and soft stops?",
                "What is the procedure for homing all axes?",
                "How to set the tool length offset?",
                "What are the alarm codes and their meanings?",
                "How to backup and restore the CNC machine configuration?",
                # Safety and consignation procedure queries
                "What are the steps to safely consign the machine?",
                "How to safely lock out the machine for maintenance?",
                "What is the consignation procedure for the CNC machine?",
                "Steps to safely shut down the machine?",
                "How to perform a lockout tagout on the machine?",
                "What are the safety steps before opening the electrical cabinet?",
                "How to safely stop the machine in an emergency?",
                "What is the procedure for resuming operation after a safety stop?",
                "What precautions are needed before performing maintenance?",
                # Description and definition queries — common KG questions
                "What is the description of the Pneumatic System?",
                "What is the description of the Hydraulic Unit?",
                "Describe the cooling system of the machine.",
                "What does the lubrication system do?",
                "Tell me about the spindle unit.",
                "What is the function of the tool magazine?",
                "Describe the electrical cabinet.",
                "What is the CNC controller?",
                "What is the Pneumatic System?",
                "What is the Hydraulic System?",
                "Describe the safety door interlock.",
                "What are the components of the machine tool?",
                "What is the role of the chip conveyor?",
            ],
            'monitoring': [
                "What is the current spindle speed?",
                "Is the CNC machine currently running?",
                "What is the current feed rate?",
                "Show me the live axis positions",
                "What is the current spindle power consumption?",
                "Is the machine in auto mode or manual mode?",
                "What tool is currently loaded?",
                "What is the active program name?",
                "Are there any active alarms on the machine?",
                "What is the current status of the CNC machine?",
                "Show me the live sensor readings from the machine",
                "What is the machine doing right now?"
            ],
            'analytics': [
                # Workpiece-level analysis
                "Analyze the machining data for workpiece OF10001",
                "Show me the spindle power trends for OF10002",
                "Compare vibration levels across workpieces OF10001 to OF10005",
                "What are the chatter detection results for OF10003?",
                "Analyze the power consumption during the last operation",
                # Time-based analysis
                "Show me the spindle speed trends from last week",
                "What were the axis power levels between September and October?",
                "Analyze the vibration severity over the past month",
                "Compare machining performance across different time periods",
                # CNC-specific patterns
                "Show me the feed rate variations during cutting",
                "Analyze the spindle load patterns for tool wear detection",
                "What are the vibration frequency patterns in the X axis?",
                "Show me the correlation between spindle speed and power consumption",
                "Analyze chatter frequency trends over time",
                "What is the average power consumption per workpiece?",
                "Show me the historical temperature trends at the spindle head"
            ]
        }
        
        # Generate embeddings for all example questions
        self.question_embeddings = {}
        self.category_mapping = {}
        
        try:
            for category, questions in self.example_questions.items():
                embeddings = []
                for question in questions:
                    embedding = self.embeddings.embed_query(question)
                    embeddings.append(embedding)
                    self.category_mapping[question] = category
                    self.question_embeddings[category] = np.array(embeddings)
        except Exception as e:
            # Fallback: use simple keyword matching if embeddings fail
            logging.error(f"Warning: Semantic classification initialization failed: {e}")
            self.question_embeddings = None

    def _cosine_similarity(self, vec1, vec2):
        """Calculate cosine similarity between two vectors."""
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    def _semantic_classify_query(self, query: str, similarity_threshold: float = 0.7) -> Dict[str, float]:
        """
        Use semantic similarity to classify query against example questions.
        
        Args:
            query: User's question
            similarity_threshold: Minimum similarity score to consider a match
            
        Returns:
            dict: Category confidence scores
        """
        if self.question_embeddings is None:
            # Fallback to equal weights if semantic search is unavailable
            return {'documentation': 0.33, 'monitoring': 0.33, 'analytics': 0.33}
        
        try:
            # Get embedding for the user query
            query_embedding = self.embeddings.embed_query(query)
            
            category_scores = {}
            
            for category, embeddings in self.question_embeddings.items():
                # Calculate cosine similarity with all examples in this category
                similarities = []
                for embedding in embeddings:
                    similarity = self._cosine_similarity(query_embedding, embedding)
                    similarities.append(similarity)
                
                # Use the maximum similarity score for this category
                max_similarity = max(similarities) if similarities else 0
                category_scores[category] = max_similarity
            
            return category_scores
            
        except Exception as e:
            logging.error(f"Warning: Semantic classification failed: {e}")
            # Fallback to equal weights
            return {'documentation': 0.33, 'monitoring': 0.33, 'analytics': 0.33}

    def initialize_conversation_chain(self, source_filter: str = None):
        """Initialize the conversation chain for the integrated agent with use case-specific context."""

        # Create context-aware system message based on source filter
        # Single CNC system prompt (source_filter kept for future use but not branching anymore)
        system_template = """
        You are an Integrated Manufacturing Assistant specialized in 5-axis CNC milling machine operations.
        You combine documentation knowledge, real-time monitoring, and historical analytics to support
        operators and engineers working with CNC workpieces (OF10001 to OF10005).

        DATA AVAILABLE:
        Each workpiece (OF10001–OF10005) has three sensor data files:
          - TYZBPS: general machine state (spindle speed, feed rate, axis positions, temperature, tool data)
          - BXCZ3M: axis power consumption (X1, X2, Y, Z axes + active/apparent/reactive power)
          - 7N4ZJ8: vibration and chatter detection (severity, frequency, chatter on/off per axis)

        YOUR CAPABILITIES:

        DOCUMENTATION & KNOWLEDGE (via knowledge graph):
        - CNC machine technical specifications and operating parameters
        - Tool change and calibration procedures
        - Safety protocols and maintenance schedules
        - G-code programming and workpiece setup instructions
        - Troubleshooting guides and alarm code explanations
        - Cutting parameter recommendations for different materials

        REAL-TIME MONITORING (via SINDIT DT Platform):
        - Live spindle speed, feed rate, and axis positions
        - Current machine status (auto/manual/alarm)
        - Active tool number and running program
        - Live power consumption per axis

        HISTORICAL ANALYTICS & INSIGHTS (via Analytics Agent):
        - Workpiece-level machining analysis (OF10001–OF10005)
        - Spindle power and axis load trends over time
        - Vibration severity and chatter frequency patterns
        - Feed rate and spindle speed variations during cutting
        - Tool wear indicators from power and vibration signals
        - Cross-workpiece comparisons and performance benchmarking

        Data time range: September 2025 – March 2026.
        Always clearly indicate which data source you are drawing from in your response.
        """

        # Common integration approach for all templates
        integration_approach = """
        INTEGRATION APPROACH:
        When answering questions:
        1. First determine if the question requires static knowledge, real-time data, historical analytics, or combinations
        2. Retrieve relevant documentation/manuals if needed for context
        3. Get current real-time data if the question involves current status or monitoring
        4. Perform historical analysis if the question involves trends, patterns, or performance optimization
        5. Combine all sources to provide comprehensive, accurate responses
        6. Clearly distinguish between documented procedures, current operational status, and historical insights
        7. Provide related information that bridges theory, current practice, and historical performance

        RESPONSE STRUCTURE:
        - Start with current status/data if relevant
        - Provide contextual documentation/procedures as needed
        - Include historical insights and trends when applicable
        - Indicate data sources (documentation vs. real-time vs. historical) for transparency

        If any data source is unavailable, clearly indicate this and work with available information.
        Always prioritize accuracy and practical utility in manufacturing contexts.
        """

        full_template = system_template + integration_approach

        system_msg = SystemMessagePromptTemplate.from_template(template=full_template)
        human_msg = HumanMessagePromptTemplate.from_template(template="{input}")

        self.prompt_template = ChatPromptTemplate.from_messages([system_msg, human_msg])

        self.conversation_chain = self.prompt_template | self.llm

    def classify_query_type(self, query: str) -> Dict[str, bool]:
        """
        Classify what type of information the query requires using semantic understanding.
        
        This simplified approach relies on:
        1. Semantic vector similarity against curated examples
        2. LLM classification for edge cases
        3. Time pattern detection for analytics override

        Args:
            query: User's question

        Returns:
            dict: Classification of query requirements
        """
        query_lower = query.lower()

        # STEP 0: Conversational / meta-query detection.
        # If the user is asking about the conversation itself (not about machines or data),
        # no external agent is needed — the LLM answers directly from the enriched context
        # that already contains the full conversation history.
        _CONVERSATIONAL_PATTERNS = [
            'what did i ask', 'what did you say', 'what was my',
            'my previous question', 'my last question',
            'your previous answer', 'your last answer',
            'what have we discussed', 'what have you told',
            'remind me what', 'summarize our conversation',
            'what was the question', 'repeat what',
            'tell me what you said', 'what did we talk',
        ]
        if any(p in query_lower for p in _CONVERSATIONAL_PATTERNS):
            # Pure conversation — all agents OFF, the LLM handles it alone
            return {'MonitoringAgent': False, 'RetrieverAgent': False, 'AnalyticsAgent': False}

        # STEP 1: Semantic vector search (primary classification method)
        semantic_scores = self._semantic_classify_query(query)
        
        # Find the most confident semantic category
        max_semantic_category = max(semantic_scores.items(), key=lambda x: x[1])
        max_semantic_score = max_semantic_category[1]
        
        # STEP 2: High confidence semantic classification
        if max_semantic_score > 0.8:
            # High confidence - use semantic classification directly
            needs_realtime = max_semantic_category[0] == 'monitoring'
            needs_documentation = max_semantic_category[0] == 'documentation'
            needs_analytics = max_semantic_category[0] == 'analytics'
            
            # For very high confidence (>0.95), skip time pattern detection to avoid overrides
            time_override_allowed = max_semantic_score <= 0.95
        else:
            # STEP 3: Low confidence - use LLM classification
            llm_classification = self._llm_classify_query(query)
            needs_realtime = llm_classification.get('realtime', False)
            needs_documentation = llm_classification.get('documentation', True)  # Default fallback
            needs_analytics = llm_classification.get('analytics', False)
            
            # Allow time pattern override for LLM classifications
            time_override_allowed = True

        # STEP 4: Time pattern detection (only for analytics enhancement)
        if time_override_allowed and self._has_time_patterns(query_lower):
            needs_analytics = True

        # STEP 4b: Workpiece guard — analytics parquet data only exists for OF10001–OF10005.
        # If analytics is triggered but:
        #   - no workpiece (OF10001–OF10005) is explicitly mentioned, AND
        #   - no strong/unambiguous time indicator is present (HH:MM, month name, "last week"…)
        # → this is almost certainly a misfired classification (conversational question,
        #   documentation query, or a machine ref that isn't a CNC workpiece).
        # Use _has_strong_time_indicator (not the broad _has_time_patterns) to avoid
        # false positives from words like "previous", " on ", " at ".
        if needs_analytics and not self._has_workpiece_reference(query_lower):
            if not self._has_strong_time_indicator(query_lower):
                needs_analytics = False
                needs_documentation = True  # Redirect to documentation

        # STEP 5: Safety net — ensure at least one agent is selected,
        # but ONLY if this isn't a conversational question (already handled in Step 0).
        # If we reach here with all False, it means the LLM classification returned all False
        # (unlikely but possible). Use the best semantic score as tiebreaker.
        if not needs_realtime and not needs_documentation and not needs_analytics:
            if max_semantic_category[0] == 'monitoring':
                needs_realtime = True
            elif max_semantic_category[0] == 'analytics':
                needs_analytics = True
            else:
                # Default to documentation — safer than triggering analytics
                needs_documentation = True

        return {
            'MonitoringAgent': bool(needs_realtime),
            'RetrieverAgent': bool(needs_documentation),
            'AnalyticsAgent': bool(needs_analytics),
            # # Include diagnostic information for debugging
            # '_classification_debug': {
            #     'semantic_scores': {k: float(v) for k, v in semantic_scores.items()},
            #     'max_semantic_category': (max_semantic_category[0], float(max_semantic_category[1])),
            #     'max_semantic_score': float(max_semantic_score),
            #     'classification_method': 'semantic' if max_semantic_score > 0.8 else 'llm',            
            #     }
        }

    def _has_time_patterns(self, query_lower: str) -> bool:
        """
        Detect time-based patterns that indicate analytics queries.
        
        Args:
            query_lower: Lowercase query string
            
        Returns:
            bool: True if time patterns are detected
        """
        time_patterns = [
            # Time range indicators
            ' from ', ' to ', ' between ', 
            # Date indicators (months)
            'december', 'january', 'february', 'march', 'april', 'may', 
            'june', 'july', 'august', 'september', 'october', 'november',
            # Time indicators  
            ' on ', ' at ', ' during ', ':', ' am', ' pm', 
            # Relative time indicators
            'yesterday', ' last ', 'previous', 'past ', 'over time',
            # Specific time phrases (precise to avoid false positives)
            'for the last ', 'for the past ', 'in the last ', 'in the past ',
            'over the last ', 'over the past ', 'during the last ', 'during the past '
        ]
        
        # Check for time patterns
        has_time_pattern = any(time_pattern in query_lower for time_pattern in time_patterns)
        
        # Check for time + duration combinations
        time_durations = ['hour', 'day', 'week', 'month', 'year', 'quarter']
        has_duration = any(duration in query_lower for duration in time_durations)
        
        return has_time_pattern or (has_duration and any(word in query_lower for word in ['last', 'past', 'previous', 'ago']))

    def _has_strong_time_indicator(self, query_lower: str) -> bool:
        """
        Return True only for unambiguous, explicit time references.

        This is a stricter subset of _has_time_patterns, used specifically in
        the workpiece guard (Step 4b). It avoids false positives from words like
        "previous", " on ", " at ", " to " that appear in everyday sentences
        but have nothing to do with data time ranges.
        """
        import re as _re
        # Month names are unambiguous
        months = [
            'january', 'february', 'march', 'april', 'may', 'june',
            'july', 'august', 'september', 'october', 'november', 'december',
        ]
        if any(m in query_lower for m in months):
            return True
        # Relative time phrases long enough to be unambiguous
        strong_phrases = [
            'yesterday',
            ' last week', ' last month', ' last year', ' last hour',
            ' last day',
            'for the last ', 'for the past ',
            'in the last ',  'in the past ',
            'over the last ', 'over the past ',
            'during the last ', 'during the past ',
        ]
        if any(p in query_lower for p in strong_phrases):
            return True
        # HH:MM pattern (a colon between digits = explicit time)
        if _re.search(r'\d{1,2}:\d{2}', query_lower):
            return True
        # Absolute date patterns like 2025-09-01
        if _re.search(r'\d{4}-\d{2}-\d{2}', query_lower):
            return True
        return False

    def _has_workpiece_reference(self, query_lower: str) -> bool:
        """
        Return True if the query explicitly mentions a CNC workpiece ID (OF10001–OF10005).

        This is used as a guard: the analytics parquet data only covers these five
        workpieces. A query that triggers analytics but doesn't mention any of them
        is almost certainly a misfired classification (e.g. a documentation or
        procedure question mentioning a machine number like "machine 11007").
        """
        workpiece_patterns = [
            'of10001', 'of10002', 'of10003', 'of10004', 'of10005',
            'workpiece', 'work order', 'work-order',
        ]
        return any(p in query_lower for p in workpiece_patterns)

    def _llm_classify_query(self, query: str) -> Dict[str, bool]:
        """
        Use LLM to classify query when keyword matching is insufficient.

        Args:
            query: User's question

        Returns:
            dict: LLM-based classification
        """
        try:
            classification_prompt = f"""
            Classify this query into the correct information source(s) for a CNC machine assistant.

            Query: "{query}"

            SOURCE DEFINITIONS (read carefully before deciding):
            - "realtime": live sensor values right now — current spindle speed, active alarms, current feed rate, machine ON/OFF status.
            - "documentation": descriptions, definitions, procedures, manuals — e.g. "what is X", "describe X", "what does X do", "how to do Y", safety rules, specifications.
            - "analytics": historical time-series analysis of recorded parquet data files (OF10001–OF10005) — trends, power charts, vibration analysis, chatter detection over time.

            IMPORTANT: a question asking "what is the description of X" or "describe X" is ALWAYS documentation, never analytics.
            Analytics is ONLY for questions about historical data trends, numerical time-series, or workpiece performance over time.

            Respond ONLY with a JSON object, no explanation:
            {{
                "realtime": true/false,
                "documentation": true/false,
                "analytics": true/false
            }}
            """

            response = self.llm.invoke(classification_prompt)

            # FIX: extract the text content before applying regex (invoke returns an object, not a string)
            response_text = response.content if hasattr(response, "content") else str(response)
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                classification = json.loads(json_match.group())
                return classification

        except Exception:
            pass

        # Fallback: documentation only
        return {"realtime": False, "documentation": True, "analytics": False}

    def query(self, user_query: str, source_filter: str = None) -> str:
        """
        Process a user query using integrated knowledge, monitoring, and analytics.

        Args:
            user_query: User's question
            source_filter: Optional source filter for documentation retrieval (e.g., 'uc1', 'uc2')

        Returns:
            str: Comprehensive integrated response
        """
        # Reinitialize conversation chain with appropriate context for the source filter
        self.initialize_conversation_chain(source_filter=source_filter)
        
        # Extract just the current question for classification (strip history preamble added by chatbot_ui)
        # The enriched query looks like: "Conversation history:\nUser: ...\nCurrent question: <actual question>"
        question_for_classification = user_query
        if "Current question:" in user_query:
            question_for_classification = user_query.split("Current question:")[-1].strip()

        # Classify the query type using only the clean question (not the full enriched string)
        query_classification = self.classify_query_type(question_for_classification)

        # Gather context based on query requirements
        context = {
            "user_query": user_query,
            "timestamp": datetime.now().isoformat(),
            "query_classification": query_classification,
            "source_filter": source_filter
        }

        logging.info("QUERY CLASSIFICATION : ")
        logging.info(query_classification)

        # CONVERSATIONAL SHORTCUT: if no agent is needed (pure meta/conversational question),
        # skip all agents and answer directly from the enriched user_query which already
        # contains the full conversation history injected by chatbot_ui.py.
        no_agent_needed = not query_classification['RetrieverAgent'] and not query_classification['MonitoringAgent'] and not query_classification['AnalyticsAgent']
        
        if no_agent_needed:
            logging.info("No agent needed — answering directly from conversation history")
            result = self.conversation_chain.invoke({"input": user_query})
            return result.content if hasattr(result, "content") else str(result)

        # Get documentation context if needed
        if query_classification['RetrieverAgent']:
            logging.info("I need the RetrieverAgent")
            try:
                doc_context = self.retriever_agent.get_documentation_context(user_query, source_filter=source_filter)
                context["documentation"] = doc_context
            except Exception as e:
                context["documentation"] = {"error": f"Documentation unavailable: {str(e)}"}

        # Get real-time context if needed (only if monitoring agent is connected)
        if query_classification['MonitoringAgent']:
            logging.info("I need the MonitoringAgent")
            if self.monitoring_agent is not None:
                try:
                    realtime_context = self.monitoring_agent.get_realtime_context(user_query)
                    context["realtime"] = realtime_context
                except Exception as e:
                    context["realtime"] = {"error": f"Real-time monitoring unavailable: {str(e)}"}
            else:
                context["realtime"] = {"error": "MonitoringAgent not connected."}

        # Get analytics context if needed
        self.last_figure = None  # Reset before each query
        if query_classification['AnalyticsAgent']:
            logging.info("I need the AnalyticsAgent")
            try:
                # Parse time range from the user query first
                parsed_time_range = None
                if hasattr(self.analytics_agent, 'parse_time_range_from_query'):
                    parsed_time_range = self.analytics_agent.parse_time_range_from_query(user_query)
                    # Store the parsed time range for potential use by other components
                    self.last_parsed_time_range = parsed_time_range

                # Pass the parsed time range to the analytics agent
                analytics_result = self.analytics_agent.query(user_query, parsed_time_range)
                context["analytics"] = analytics_result

                # Propagate the last Plotly figure so chatbot_ui can display it
                self.last_figure = self.analytics_agent.last_figure
            except Exception as e:
                context["analytics"] = {"error": f"Analytics unavailable: {str(e)}"}

            except Exception as e:
                return {"error": f"Workpiece metrics failed: {str(e)}"}

        # Format the input for the conversation chain - handle datetime and other serialization
        def json_serialize_helper(obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            elif isinstance(obj, (bool, int, float, str, list, dict)):
                return obj
            return str(obj)

        try:
            context_json = json.dumps(context, indent=2, default=json_serialize_helper)
            classification_json = json.dumps(query_classification, indent=2, default=json_serialize_helper)
        except Exception as e:
            # Fallback to string representation if JSON serialization fails
            context_json = str(context)
            classification_json = str(query_classification)

        formatted_input = f"""
        User Query: {user_query}

        Query Classification:
        {classification_json}

        Available Context:
        {context_json}

        Please provide a comprehensive response that integrates documentation knowledge,
        real-time monitoring data, and historical analytics as appropriate for this query.
        """

        result = self.conversation_chain.invoke({"input": formatted_input})
        return result.content if hasattr(result, "content") else str(result)

    def get_system_status(self) -> Dict[str, Any]:
        """
        Get comprehensive system status combining all data sources.

        Returns:
            dict: Complete system status
        """
        status = {
            "system_components": {}
        }

        # Get monitoring status (optional — only if connected)
        if self.monitoring_agent is not None:
            try:
                monitoring_overview = self.monitoring_agent.get_sindit_status()
                status["system_components"]["monitoring"] = monitoring_overview
            except Exception as e:
                status["system_components"]["monitoring"] = {"status": "error", "error": str(e)}
        else:
            status["system_components"]["monitoring"] = {"status": "not connected"}

        try:
            available_sources = self.retriever_agent.get_available_sources()
            status["system_components"]["documentation"] = {
                "status": "available",
                "sources": available_sources
            }
        except Exception as e:
            status["system_components"]["documentation"] = {"status": "error", "error": str(e)}

        # Get analytics system status
        try:
            analytics_status = {
                "status": "available",
                "loaded_data": list(self.analytics_agent.loaded_data.keys()) if hasattr(self.analytics_agent, 'loaded_data') else [],
                "default_time_range": str(self.analytics_agent.default_time_range) if hasattr(self.analytics_agent, 'default_time_range') else "unknown"
            }
        except Exception as e:
            analytics_status = {"status": "error", "error": str(e)}
        status["system_components"]["analytics"] = analytics_status

        return status