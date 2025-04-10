import argparse
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM
from collections import defaultdict
from config_loader import load_config

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

config = load_config()

def get_embedding_function():
    try:
        model_name = config.get("model", "mistral")
        return OllamaEmbeddings(model=model_name, temperature=0)
    except ConnectionError:
        print("Failed to connect to Ollama. Please check the service and try again.")
        return None


def query_rag_per_doc(query_text: str):
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    results = db.similarity_search_with_score(query_text, k=20)

    # Group results by document (source)
    grouped_contexts = defaultdict(list)
    for doc, score in results:
        source = doc.metadata.get("source", "unknown")
        grouped_contexts[source].append(doc.page_content)

    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

    model_name = config.get("model", "mistral")
    model = OllamaLLM(model=model_name, temperature=0.5)

    final_results = {}

    for source, pages in grouped_contexts.items():
        print("Query source:", source)
        context_text = "\n\n---\n\n".join(pages)
        prompt = prompt_template.format(context=context_text, question=query_text)
        response_text = model.invoke(prompt)
        final_results[source] = response_text

    for source, response in final_results.items():
        print(f"\n=== Answer from {source} ===\n{response}\n")

    return final_results

def query_rag(query_text: str):
    # Prepare the DB.
    embedding_function = get_embedding_function()

    print("Prepare db connection")
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB.
    print("Searching in db...")
    results = db.similarity_search_with_score(query_text, k=8)
    print("Found result", len(results))

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

    print("Create prompt")
    prompt = prompt_template.format(context=context_text, question=query_text)

    print("Run model")
    model = OllamaLLM(model="mistral")

    print("Model started - run prompt")
    response_text = model.invoke(prompt)

    print("Got response")
    sources = [doc.metadata.get("id", None) for doc, _score in results]

    print("Got sources")
    formatted_response = f"Response: {response_text}\nSources: {sources}"

    print(formatted_response)
    return response_text


if __name__ == "__main__":
    # Create CLI.
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text
    query_rag_per_doc(args.query_text)
    # query_rag(args.query_text)
