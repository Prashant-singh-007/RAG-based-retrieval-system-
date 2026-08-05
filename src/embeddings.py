from langchain_ollama import OllamaEmbeddings
from langchain_core.embeddings import Embeddings

# Paper (Section 3.1): "Each resulting text chunk is converted into a vector
# embedding using nomic-embed-text v1.5 [20]." We run it locally via Ollama.
MODEL_NAME = "nomic-embed-text"


class NomicEmbeddings(Embeddings):
    """Wraps nomic-embed-text v1.5 (served locally via Ollama) with the
    task-specific prefixes it expects for asymmetric document/query retrieval.
    Without these prefixes, retrieval quality with this model degrades
    noticeably, since it was trained with them.
    """

    def __init__(self, model_name: str = MODEL_NAME):
        self._model = OllamaEmbeddings(model=model_name)

    def embed_documents(self, texts):
        return self._model.embed_documents([f"search_document: {t}" for t in texts])

    def embed_query(self, text):
        return self._model.embed_query(f"search_query: {text}")


def get_embedding_model():
    return NomicEmbeddings()
