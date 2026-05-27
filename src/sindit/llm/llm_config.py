import os
# from typing import Any, Optional
from dotenv import load_dotenv

load_dotenv()
CHAT_MODEL = os.getenv("OLLAMA_MODEL")
TEMPERATURE = os.getenv("TEMPERATURE", 0.0)

# def get_llm_instance(
#     model_name: Optional[str] = None,
#     temperature: float = 0.0,
#     max_tokens: Optional[int] = None,
#     **kwargs
# ) -> Any:
#     """
#     Args:
#         model_name: Model name (if None, uses CHAT_MODEL from env)
#         temperature: Generation temperature
#         max_tokens: Maximum tokens to generate
#         **kwargs: Additional model-specific parameters
    
#     Returns:
#         ChatOllama LLM instance
#     """
    
#     # Get model from env if not provided
#     if model_name is None:
#         model_name = os.getenv('OLLAMA_MODEL', 'qwen3:1.7b')
    
#     # Clean model name
#     model_name = model_name.strip('"')
    
#     print(f"🤖 Initializing LLM: {model_name}")
    
#     try:
#         from langchain_ollama import ChatOllama
        
#         print(f"🦙 Using Ollama model: {model_name}")
        
#         llm = ChatOllama(
#             model=model_name,
#             temperature=temperature,
#             base_url=os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434'),
#             # think=True,
#             # stop=["\n\n", "**", "Explanation", "Note"],
#             **kwargs
#         )
        
#         # Connection test — stop immediately if Ollama is unavailable
#         try:
#             response = llm.invoke("Hello")  # noqa: F841
#             print(f"✅ Ollama connection successful")
#         except Exception as e:
#             raise ConnectionError(
#                 f"❌ Ollama inaccessible ({e})\n"
#                 f"   → StartTom the SSH tunnel : ssh -L 11434:localhost:11434 exemple@mainframe.sintef.no\n"
#                 f"   → Or run Ollama locally : ollama serve"
#             ) from e

#         return llm
        
#     except ImportError:
#         print("❌ langchain-ollama not installed. Install with: pip install langchain-ollama")
#         raise
        
        
_llm = None  # Lazy init — created on the first call


def _get_llm():
    global _llm
    if _llm is None:
        from langchain_ollama import ChatOllama
        import logging
        model_name = CHAT_MODEL.strip('"') if CHAT_MODEL else 'qwen3:8b-16k'
        logging.info(f"🤖 Initializing LLM: {model_name}")
        _llm = ChatOllama(
            model=model_name,
            temperature=0.0,
            base_url=os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434'),
            think=False,  # Disables qwen3's thinking mode
            timeout=120,  # 2-minute hard timeout — prevents silent hangs
        )
        logging.info(f"✅ LLM initialized")
    return _llm
