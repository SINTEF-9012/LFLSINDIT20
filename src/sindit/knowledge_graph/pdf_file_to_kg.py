import logging
import os


# Print startup message BEFORE heavy imports (langchain, pypdf, etc.)
if __name__ == "__main__":
    logging.info("Loading the app!")

from .pdf_processing import (
    load_and_chunk_pdf,
    process_pdf_directory_individually,
    process_subdirectories_individually
)
from .graph_transformer import extract_and_store_graph

from tqdm import tqdm

def process_pdf_file(pdf_path: str):
    """
    Process a single PDF file, extract its content, and create a graph.
    
    Args:
        pdf_path (str): Path to the PDF file.
    """
    logging.info("process_pdf_file !!!\n")
    # Check if the file exists
    if not os.path.isfile(pdf_path):
        raise ValueError(f"File path {pdf_path} is not a valid file.")

    # Load and chunk the PDF file
    logging.info(f"Loading and chunking PDF file: {pdf_path}")
    documents = load_and_chunk_pdf(pdf_path)

    # Process documents and create graph
    process_documents_to_graph(documents)

def process_pdf_directory_individual(directory_path: str):
    """
    Process all PDF files in a directory individually.
    
    Args:
        directory_path (str): Path to the directory containing PDF files.
    """
    logging.info(f"Processing all PDF files in directory: {directory_path}")
    documents, file_mapping = process_pdf_directory_individually(directory_path)
    
    logging.info(f"File mapping: {file_mapping}")
    process_documents_to_graph(documents)

def process_subdirectories_with_separate_graphs(directory_path: str):
    """
    Process subdirectories individually, creating separate graphs for each subdirectory.
    
    Args:
        directory_path (str): Path to the directory containing subdirectories with PDF files.
    """
    logging.info(f"Processing subdirectories in: {directory_path}")
    subdirectory_results = process_subdirectories_individually(directory_path)
    
    if not subdirectory_results:
        logging.info("No subdirectories found or processed.")
        return
    
    # Collect processed sources
    processed_sources = []
    
    # Process each subdirectory's documents separately
    for subdir_name, result in subdirectory_results.items():
        if 'error' in result:
            logging.error(f"Skipping {subdir_name} due to error: {result['error']}")
            continue
            
        documents = result['documents']
        file_mapping = result['file_mapping']
        
        if not documents:
            logging.info(f"No documents found in subdirectory: {subdir_name}")
            continue
        
        logging.info(f"\n{'='*50}")
        logging.info(f"Creating graph for subdirectory: {subdir_name}")
        logging.info(f"Documents to process: {len(documents)}")
        logging.info(f"File mapping: {file_mapping}")
        logging.info(f"{'='*50}")
        
        # Create a separate graph for this subdirectory
        process_documents_to_graph(documents, subdir_name)
        processed_sources.append(subdir_name)
    
    return processed_sources

def process_documents_to_graph(
    documents,
    graph_name: str = "default",
    start_chunk: int = 0,
    clean_graph: bool = False,
):
    """
    Process document chunks and create a graph.

    Args:
        documents   : List of document chunks to process.
        graph_name  : Name identifier for the graph (default: "default").
        start_chunk : Index of the first chunk to process (0-based).
                      Chunks before this index are skipped — useful for
                      resuming after a crash without reprocessing everything.
                      (default: 0 — process all chunks)
        clean_graph : If True, wipe the existing Knowledge Graph before
                      starting. Set to False when resuming from a crash so
                      that already-stored assets are preserved. (default: False)
    """
    logging.info(f"Processing document chunks for graph: {graph_name}")

    from ..util.sindit_client import SINDITClient
    client = SINDITClient()

    if clean_graph:
        logging.info("[KG] Cleaning existing graph before processing...")
        client.clean_graph()
        logging.info("[KG] Graph cleaned.")
    else:
        logging.warning("[KG] Skipping graph clean (clean_graph=False).")

    assets = []
    # dict id → {"id", "type", "page"} — keeps the first occurrence per asset
    distinct_assets: dict = {}
    relations = []

    total = len(documents)

    if start_chunk > 0:
        logging.warning(f"[KG] Resuming from chunk {start_chunk + 1}/{total} (skipping first {start_chunk} chunk(s)).")

    effective_docs = documents[start_chunk:]
    logging.info(f"[KG] Chunks to process: {len(effective_docs)} (total in file: {total})")

    for i, doc in enumerate(tqdm(effective_docs, desc=f"Processing documents for {graph_name}\n")):
        absolute_index = start_chunk + i
        page_num = doc.metadata.get('page', '?')
        logging.info(f"[KG] Chunk {absolute_index + 1}/{total} — {len(doc.page_content)} chars (page {page_num})")
        graph_document = extract_and_store_graph(doc, source_label=graph_name)
        if graph_document:
            assets.extend(graph_document.assets or [])
            for asset in graph_document.assets:
                if asset.id not in distinct_assets:
                    # Extract the page number from the properties added by graph_transformer
                    page_prop = next(
                        (p.propertyValue for p in (asset.properties or []) if p.propertyName == "page"),
                        str(page_num)
                    )
                    distinct_assets[asset.id] = {
                        "id": asset.id,
                        "type": asset.assetType or "",
                        "page": page_prop,
                    }

            for relation in graph_document.relationships:
                relations.append([relation.sourceId, relation.targetId, relation.relationshipType])

    asset_list = list(distinct_assets.values())

    logging.info(f"Graph '{graph_name}' - Distinct assets: {len(asset_list)}")
    logging.info(f"Graph '{graph_name}' - Relations: {len(relations)}")
    logging.info(f"Graph '{graph_name}' - Completed processing")
    logging.info("All assets:")
    for a in asset_list:
        logging.info(f"- id={a['id']}, type={a['type']}, page={a['page']}")

    return {
        'graph_name': graph_name,
        'asset_count': len(asset_list),
        'relation_count': len(relations),
        'distinct_assets': asset_list,   # liste de {id, type, page}
        'relations': relations
    }

    
if __name__ == "__main__":
    # ─────────────────────────────────────────────────────────────────
    # CONFIGURATION — edit these variables before running
    # ─────────────────────────────────────────────────────────────────

    # Path to the PDF file to process
    PDF_PATH = os.path.join(os.getcwd(), "data", "documents", "DOCUMENTACIÓN DE LA MÁQUINA 11007.pdf")

    # Skip the first N pages of the PDF (0 = process all pages).
    # Useful to skip cover page, table of contents, legal notices, etc.
    # Example: START_PAGE = 5  →  processing starts at page index 5
    START_PAGE = 260

    # Skip the first N chunks (0 = process all chunks).
    # Use this to RESUME after a crash: set to the last successfully
    # processed chunk index so you don't redo the work already stored in the KG.
    # Example: START_CHUNK = 49  →  resumes at chunk 50
    START_CHUNK = 0

    # Set to True to wipe the Knowledge Graph before starting.
    # Set to False when resuming from a crash (keeps already-stored data).
    CLEAN_GRAPH = False

    # Name used to tag all assets and relations in the graph
    GRAPH_NAME = "default"

    # Chunking parameters
    CHUNK_SIZE = 2000  # tokens per chunk (approx)
    CHUNK_OVERLAP = 100   # overlap between chunks

    # ─────────────────────────────────────────────────────────────────

    PDF_PATH = os.path.normpath(PDF_PATH)
    logging.info("Imports done — app ready!")

    # Check Ollama connection before processing
    from .graph_transformer import _get_llm
    try:
        _get_llm()
    except ConnectionError as e:
        logging.info(e)
        exit(1)

    logging.info(f"Current directory: {os.getcwd()}")
    logging.info(f"PDF path: {PDF_PATH}")
    logging.info(f"File found: {os.path.isfile(PDF_PATH)}")
    logging.info(f"Start page: {START_PAGE}")
    logging.info(f"Start chunk: {START_CHUNK}")
    logging.info(f"Clean graph: {CLEAN_GRAPH}")

    if os.path.isfile(PDF_PATH):
        logging.info("Loading and chunking the PDF...")
        documents = load_and_chunk_pdf(PDF_PATH, CHUNK_SIZE, CHUNK_OVERLAP, start_page=START_PAGE)
        logging.info(f"Total chunks after filtering: {len(documents)}")
        process_documents_to_graph(documents, graph_name=GRAPH_NAME, start_chunk=START_CHUNK, clean_graph=CLEAN_GRAPH)
        logging.info("KG has been successfully established!")