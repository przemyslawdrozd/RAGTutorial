import yaml
from langchain_ollama import OllamaLLM, OllamaEmbeddings

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

\"\"\"
{context}
\"\"\"

---

Question: {question}
Answer:
"""


def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


EMBEDDING_MODEL = load_config().get("embedding_model", "mistral")
LLM_MODEL = load_config().get("llm_model", "mistral")

def load_embedding_model():
    return OllamaEmbeddings(model=EMBEDDING_MODEL)


def load_llm_model():
    return OllamaLLM(model=LLM_MODEL, temperature=0.5)


def filter_results_by_score(results, threshold=0.85):
    return [(doc, score) for doc, score in results if score < threshold]
