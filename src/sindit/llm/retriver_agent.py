import json
import logging
from .llm_config import _get_llm

logger = logging.getLogger(__name__)

from typing import Dict, List

from ..knowledge_graph.sindit_graph_tool import SINDITGraphTool

from langchain_core.callbacks import BaseCallbackHandler

class DebugPromptCallback(BaseCallbackHandler):
    """Prints the full prompt sent to the LLM (for debugging)."""
    def on_llm_start(self, serialized, prompts, **kwargs):
        for i, p in enumerate(prompts):
            print(f"\n{'='*60}")
            print(f"[LLM PROMPT #{i+1}]")
            print(p)
            print('='*60)


class RetrieverAgent:

    def __init__(self):

        # SINDITGraphTool handles the full Text2SPARQL pipeline:
        #   user question → SPARQL query → /kg/sparql → QA answer
        self.graph_tool = SINDITGraphTool()

        # LLM instance (shared, but SINDITGraphTool also instantiates its own)
        self.llm = _get_llm()

    def classify_documentation_query(self, query: str) -> Dict[str, bool]:
        """
        Classify whether a query requires documentation retrieval.

        Args:
            query: User's question.

        Returns:
            dict: Classification flags for the query.
        """
        query_lower = query.lower()

        documentation_keywords = [
            'how to', 'setup', 'configure', 'install', 'procedure', 'manual',
            'guide', 'steps', 'tutorial', 'programming', 'troubleshoot',
            'documentation', 'explain', 'what is', 'describe'
        ]

        component_keywords = [
            # UC1 — Fischertechnik Training Factory
            'plc', 'tia portal', 'simatic', 's7-1500', 'txt controller',
            'mqtt', 'opc ua', 'node-red', 'iot gateway', 'dashboard',
            'fischertechnik', 'training factory', 'high-bay warehouse',
            'suction gripper', 'sorting line', 'vacuum gripper', 'vgr',
            'hbw', 'sld', 'mpo', 'dso', 'dsi', 'factory components',

            # UC2 — Industrial Milling Machines
            'machine 7152', 'machine 11007', 'pm-10005', 'milling machine',
            'industrial machine', 'machine operation', 'machine programming',
            'machine maintenance', 'machine safety', 'machine specifications',
            'cutting parameters', 'tool change', 'calibration', 'cnc',
            'machining', 'manufacturing equipment', 'industrial equipment'
        ]

        needs_documentation = any(keyword in query_lower for keyword in documentation_keywords)
        is_component_query = any(keyword in query_lower for keyword in component_keywords)

        return {
            'needs_documentation': needs_documentation or is_component_query,
            'is_how_to_query': 'how to' in query_lower or 'how do' in query_lower,
            'is_component_query': is_component_query,
            'is_setup_query': any(word in query_lower for word in ['setup', 'configure', 'install']),
            'is_troubleshoot_query': 'troubleshoot' in query_lower or 'problem' in query_lower
        }

    def get_documentation_context(self, user_query: str, source_filter: str = None) -> dict:
        """
        Query the knowledge graph and return structured context for the given question.

        The Text2SPARQL pipeline (graph_tool._run) already:
          1. Checks conversation history — returns early if the answer is already there.
          2. Has the LLM generate a SPARQL query specific to the user's question.
          3. Executes the query against GraphDB.
          4. Has the LLM format the results into a natural-language answer.

        Returns a dict with:
          - answer        : natural-language answer produced by the Text2SPARQL pipeline
          - source_filter : the filter that was applied (or None)
        """
        try:
            needs_kg, direct_answer = self.graph_tool._can_answer_from_history(user_query)
            if not needs_kg and direct_answer:
                # The answer is already in the conversation history — no KG call needed
                answer = direct_answer
            else:
                answer = self.graph_tool._run(question=user_query, source_filter=source_filter)
        except Exception as exc:
            answer = f"Knowledge graph unavailable: {exc}"

        return {
            "answer":        answer,
            "source_filter": source_filter,
        }

    def query(self, user_query: str, source_filter: str = None) -> str:
        """
        Process a user query using the Text2SPARQL pipeline.

        Flow:
          1. SINDITGraphTool generates a SPARQL query from the user's question.
          2. The query is executed via POST /kg/sparql on the SINDIT REST API.
          3. The LLM formats the SPARQL results into a natural language answer (QA).

        Args:
            user_query: User's natural language question.
            source_filter: Optional source filter (e.g. 'uc1', 'uc2').

        Returns:
            str: Natural language answer grounded in the knowledge graph.
        """
        try:
            response = self.graph_tool._run(question=user_query, source_filter=source_filter)
            return response

        except Exception as e:
            return f"Error processing query: {str(e)}"

    def get_available_sources(self) -> List[str]:
        """
        Get the list of distinct source values stored in the knowledge graph.

        Returns:
            List[str]: List of source labels (e.g. document names).
        """
        try:
            sparql = """
            PREFIX sindit: <http://sindit.sintef.no/2.0#>
            SELECT DISTINCT ?source WHERE {
                ?prop sindit:propertyName "source" ;
                      sindit:propertyValue ?source .
            }
            ORDER BY ?source
            """
            result_str = self.graph_tool._execute_sparql(sparql)
            data = json.loads(result_str)
            bindings = data.get("results", {}).get("bindings", [])
            return [b["source"]["value"] for b in bindings if "source" in b]
        except Exception as e:
            print(f"Error retrieving available sources: {e}")
            return []
