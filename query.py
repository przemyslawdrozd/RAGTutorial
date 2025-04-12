import argparse
from langchain_ollama import OllamaEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
from collections import defaultdict
from query_rag_utils import (
    CHROMA_PATH,
    PROMPT_TEMPLATE,
    load_config,
    load_embedding_model,
    load_llm_model,
    filter_results_by_score,
)

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

EMBEDDING_MODEL = load_config().get("embedding_model", "mistral")
SIMILARITY_THRESHOLD = load_config().get("similarity_threshold", 0.85)
MULTIPLE_RESPONSES = load_config().get("multiple_responses", False)


def get_embedding_function():
    try:
        return OllamaEmbeddings(model=EMBEDDING_MODEL, temperature=0)
    except ConnectionError:
        print("Failed to connect to Ollama. Please check the service and try again.")
        return None


def query_rag(query_text: str, k=8):
    print("Loading embedding model...")
    embedding_function = load_embedding_model()

    vec = embedding_function.embed_query("test")
    print("Embedding dimension:", len(vec))

    print("Connecting to Chroma...")
    db = Chroma(
        persist_directory=f"{CHROMA_PATH}/{EMBEDDING_MODEL}",
        embedding_function=embedding_function
    )

    print(f"Searching with k={k}...")
    results = db.similarity_search_with_score(query_text, k=k)
    print(f"Found {len(results)} results.")

    for doc, score in results:
        print(f"[score: {score:.4f}] {doc.metadata.get('id')} — {doc.page_content[:80]}...")

    filtered = filter_results_by_score(results, threshold=SIMILARITY_THRESHOLD)
    print(f"{len(filtered)} passed the similarity threshold.")

    if not filtered:
        print("⚠️ No high-confidence chunks found, falling back to top 3...")
        filtered = results[:3]

    model = load_llm_model()
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

    if MULTIPLE_RESPONSES:
        # Grouped per document
        grouped = defaultdict(list)
        for doc, _ in filtered:
            source = doc.metadata.get("source", "unknown")
            grouped[source].append(doc.page_content)

        final_responses = {}
        for source, pages in grouped.items():
            print("Querying for source:", source)
            context_text = "\n\n---\n\n".join(pages)
            prompt = prompt_template.format(context=context_text, question=query_text)
            response = model.invoke(prompt)
            print("Got response for source", source)
            final_responses[source] = response

        for source, response in final_responses.items():
            print(f"\n=== Answer from {source} ===\n{response}\n")

        return final_responses

    else:
        # Combined result
        context_text = "\n\n---\n\n".join([doc.page_content for doc, _ in filtered])
        prompt = prompt_template.format(context=context_text, question=query_text)
        response = model.invoke(prompt)
        sources = [doc.metadata.get("id") for doc, _ in filtered]

        print("Sources:", sources)
        print("\n=== Final Answer ===")
        print(response)
        return response


if __name__ == "__main__":
    # Create CLI.
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text
    query_rag(args.query_text)
