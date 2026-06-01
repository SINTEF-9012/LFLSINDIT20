"""SINDITGraphTool — Text2SPARQL retriever for the SINDIT GraphDB knowledge graph.

Pipeline:
  1. [SPARQL generation] LLM converts the user's question into a SPARQL SELECT query.
  2. [Execution]         The query is sent to POST /kg/sparql on the SINDIT REST API.
  3. [QA]               LLM formats the raw SPARQL results into a natural language answer.

This replaces the old "fetch all nodes then RAG" approach with a proper
Text2SPARQL flow, equivalent to GraphCypherQAChain but for GraphDB/SPARQL.
"""

import logging
import json
import os
from typing import Optional, Type

from langchain.tools import BaseTool
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel

from ..llm.llm_config import _get_llm
from ..util.sindit_client import SINDITClient

# ---------------------------------------------------------------------------
# SINDIT uses TWO distinct namespaces:
#   - GRAPH_MODEL (SAMM): urn:samm:sindit.sintef.no:1.0.0#
#     → used for RDF class types (AbstractAsset, AbstractAssetProperty...)
#     → used for all predicates (assetDescription, assetType, assetProperties,
#       propertyName, propertyValue, propertyUnit...)
#   - KG_NS: http://sindit.sintef.no/2.0#
#     → used ONLY as the base URI for individual instances (nodes/assets)
#   - label is stored with rdfs:label (W3C standard)
# ---------------------------------------------------------------------------
KG_NS       = "http://sindit.sintef.no/2.0#"
SINDIT_MODEL_NS = "urn:samm:sindit.sintef.no:1.0.0#"

# ---------------------------------------------------------------------------
# Step 1 — SPARQL generation prompt
# The LLM receives the user's question and must output a valid SPARQL query.
# ---------------------------------------------------------------------------
SPARQL_GENERATION_TEMPLATE = """You are an expert in SPARQL and RDF knowledge graphs.
Generate a single SPARQL SELECT query to answer the question below.

The graph uses TWO namespaces — you MUST use both correctly:

  PREFIX sindit:  <urn:samm:sindit.sintef.no:1.0.0#>   ← for RDF types AND predicates
  PREFIX kg:      <http://sindit.sintef.no/2.0#>         ← for instance URIs only
  PREFIX rdfs:    <http://www.w3.org/2000/01/rdf-schema#>

Common graph patterns (copy these exactly):
  ?asset a sindit:AbstractAsset .
  ?asset rdfs:label ?label .
  ?asset sindit:assetType ?type .
  ?asset sindit:assetDescription ?description .
  ?asset sindit:assetProperties ?prop .
  ?prop  sindit:propertyName  ?pname .
  ?prop  sindit:propertyValue ?pval .
  ?prop  sindit:propertyUnit  ?punit .

{source_instruction}

Question: {question}

Rules:
- Output ONLY the SPARQL query, no explanation, no markdown fences.
- Use SELECT (not CONSTRUCT or ASK).
- Limit results to 20 rows maximum.
- Always include the three PREFIX lines above at the top.
- Use rdfs:label (NOT sindit:label) for the asset name/label.
- Use sindit: prefix for all class types and predicates.
"""

SPARQL_GENERATION_PROMPT = PromptTemplate(
    input_variables=["question", "source_instruction"],
    template=SPARQL_GENERATION_TEMPLATE,
)

# ---------------------------------------------------------------------------
# Step 3 — QA (Question Answering) prompt
# The LLM receives the raw SPARQL results and must format a natural language answer.
# ---------------------------------------------------------------------------
QA_TEMPLATE = """You are an expert assistant for industrial machine documentation and knowledge.
The context below contains results from a SPARQL query executed on the SINDIT knowledge graph.

IMPORTANT RULES:
- Answer based strictly on the context provided.
- If a value is missing or null, say "not specified".
- If the context is completely empty, say "No data found in the knowledge graph."
- Be concise and technically accurate.

SPARQL results:
{context}

Question: {question}

Answer:"""

QA_PROMPT = PromptTemplate(input_variables=["context", "question"], template=QA_TEMPLATE)


# ---------------------------------------------------------------------------
# Input schema
# ---------------------------------------------------------------------------
class GraphQuestion(BaseModel):
    question: str
    source_filter: Optional[str] = None


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------
class SINDITGraphTool(BaseTool):
    """LangChain tool that queries the SINDIT GraphDB knowledge graph via Text2SPARQL.

    Flow:
      1. LLM generates a SPARQL SELECT query from the user's question.
      2. Query is executed via POST /kg/sparql on the SINDIT REST API.
      3. LLM formats the raw results into a natural language answer (QA step).
    """

    name: str = "sindit_graph_tool"
    description: str = (
        "Tool for retrieving structured knowledge about industrial assets, "
        "their properties and relationships from the SINDIT knowledge graph (GraphDB/RDF). "
        "Accepts natural language questions and returns factual answers."
    )
    args_schema: Type[GraphQuestion] = GraphQuestion

    class Config:
        arbitrary_types_allowed = True

    # ------------------------------------------------------------------
    # Step 1: Generate the SPARQL query from the user's question
    # ------------------------------------------------------------------
    def _generate_sparql(self, question: str, source_filter: Optional[str]) -> str:
        """Ask the LLM to generate a SPARQL SELECT query for the given question."""
        source_instruction = ""
        if source_filter:
            source_instruction = (
                f"Filter results: only return assets where the propertyValue of "
                f"a property named 'source' contains '{source_filter}'."
            )

        llm = _get_llm()
        prompt = SPARQL_GENERATION_PROMPT.format(
            question=question,
            source_instruction=source_instruction,
        )
        response = llm.invoke(prompt)
        sparql_query = response.content if hasattr(response, "content") else str(response)

        # Strip markdown fences if the LLM adds them
        sparql_query = sparql_query.strip()
        if sparql_query.startswith("```"):
            lines = sparql_query.split("\n")
            sparql_query = "\n".join(
                line for line in lines
                if not line.strip().startswith("```")
            )

        logging.info(f"[SINDITGraphTool] Generated SPARQL:\n{sparql_query}")
        return sparql_query.strip()

    # ------------------------------------------------------------------
    # Step 2: Execute the SPARQL query via POST /kg/sparql
    # ------------------------------------------------------------------
    def _inject_prefixes(self, sparql_query: str) -> str:
        """Guarantee the required PREFIX declarations are at the top of the query.

        The LLM sometimes puts PREFIX lines inside the WHERE block, which is
        invalid SPARQL and causes a MALFORMED QUERY error in GraphDB.
        This method:
          1. Strips every PREFIX line from wherever it appears in the query.
          2. Re-injects the mandatory prefixes at the very top.
        """
        mandatory_prefixes = [
            f"PREFIX sindit: <{SINDIT_MODEL_NS}>",
            f"PREFIX kg:     <{KG_NS}>",
            "PREFIX rdfs:   <http://www.w3.org/2000/01/rdf-schema#>",
        ]

        # Remove any existing PREFIX lines (they may be misplaced inside WHERE {})
        cleaned_lines = [
            line for line in sparql_query.splitlines()
            if not line.strip().upper().startswith("PREFIX")
        ]
        cleaned_query = "\n".join(cleaned_lines).strip()

        # Fix common LLM typos in the query body (e.g. sind3it: instead of sindit:)
        # These typos appear inside the WHERE block, not as PREFIX declarations
        for wrong in ("sind3it:", "sindit3:", "sindIt:", "sindIT:", "SINDIT:"):
            cleaned_query = cleaned_query.replace(wrong, "sindit:")

        # Re-inject all mandatory prefixes at the top
        return "\n".join(mandatory_prefixes) + "\n" + cleaned_query

    def _execute_sparql(self, sparql_query: str) -> str:
        """Send the SPARQL query to SINDIT and return the raw JSON result as text."""
        try:
            # Always guarantee the prefix is present regardless of what the LLM produced
            sparql_query = self._inject_prefixes(sparql_query)
            logging.info(f"[SINDITGraphTool] Sending SPARQL:\n{sparql_query}")

            client = SINDITClient()
            data = client.post(
                "kg/sparql",
                {
                    "query": sparql_query,
                    "accept_content": "application/sparql-results+json",
                },
            )
            result = data.get("result", data)
            return json.dumps(result, indent=2, ensure_ascii=False)
        except Exception as e:
            logging.error(f"[SINDITGraphTool] SPARQL execution failed: {e}")
            return f"SPARQL execution error: {e}"

    # ------------------------------------------------------------------
    # Step 0: Classify — can the question be answered from history?
    # ------------------------------------------------------------------
    def _can_answer_from_history(self, question: str) -> tuple:
        """Check whether the LLM can answer the question without querying the
            knowledge graph, based on the question alone.

            Note: despite the name, no actual conversation history is injected —
            the LLM only receives the current question and decides whether a KG
            lookup is necessary.

            Returns:
                (needs_kg: bool, direct_answer: str | None)
                - needs_kg=False → direct_answer contains the LLM-generated answer,
                                SPARQL query is skipped.
                - needs_kg=True  → direct_answer is None, proceed with SPARQL pipeline.
            """
        classification_prompt = f"""You are a routing assistant. Your only job is to decide if the question below can be answered using the conversation history provided, or if it requires querying a knowledge graph for new information.

{question}

Answer with valid JSON only, no explanation:
- If the answer is already in the conversation history above:
  {{"needs_kg": false, "direct_answer": "your answer here"}}
- If the question requires new information from a knowledge graph:
  {{"needs_kg": true, "direct_answer": null}}"""

        llm = _get_llm()
        response = llm.invoke(classification_prompt)
        raw = response.content if hasattr(response, "content") else str(response)

        # Strip markdown fences if the LLM wraps JSON in ```
        raw = raw.strip()
        if raw.startswith("```"):
            lines = raw.split("\n")
            raw = "\n".join(l for l in lines if not l.strip().startswith("```"))

        try:
            result = json.loads(raw.strip())
            needs_kg = result.get("needs_kg", True)
            direct_answer = result.get("direct_answer", None)
            logging.info(f"[SINDITGraphTool] History classifier → needs_kg={needs_kg}")
            return (needs_kg, direct_answer)
        except :
            # If parsing fails, default to querying the KG (safe fallback)
            logging.error(f"[SINDITGraphTool] History classifier JSON parse failed, defaulting to KG query")
            return (True, None)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------
    def _run(self, question: str, source_filter: Optional[str] = None) -> str:
        """Answer a natural language question using the SINDIT knowledge graph.

        Flow:
          Step 0 — Classify: can the question be answered from conversation history?
                   If yes → return the direct answer immediately (no SPARQL).
                   If no  → proceed with the Text2SPARQL pipeline.
          Step 1 — Generate a SPARQL query from the question.
          Step 2 — Execute the query via POST /kg/sparql.
          Step 3 — QA: format the SPARQL results into a natural language answer.

        Args:
            question: The user's natural language question (may include history prefix).
            source_filter: Optional filter (e.g. document name or source label).

        Returns:
            str: Natural language answer.
        """
        logging.info(f"[SINDITGraphTool] Question: {question}")

        # Step 0 — History classifier
        needs_kg, direct_answer = self._can_answer_from_history(question)
        if not needs_kg and direct_answer:
            logging.warning(f"[SINDITGraphTool] Answered from history, skipping SPARQL.")
            return direct_answer

        # Step 1 — Generate SPARQL
        sparql_query = self._generate_sparql(question, source_filter)

        # Step 2 — Execute query
        sparql_results = self._execute_sparql(sparql_query)
        logging.info(f"[SINDITGraphTool] SPARQL results : {sparql_results}")

        # Step 3 — QA: format results into a natural language answer
        llm = _get_llm()
        qa_prompt = QA_PROMPT.format(context=sparql_results, question=question)
        # logging.info(f"[SINDITGraphTool] QA_Prompt : {qa_prompt}")

        response = llm.invoke(qa_prompt)
        logging.info(f"[SINDITGraphTool] Response : {response}")

        return response.content if hasattr(response, "content") else str(response)
