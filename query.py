import argparse
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM

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


def get_embedding_function():
    try:
        model = "llama2:latest"
        return OllamaEmbeddings(model=model)
    except ConnectionError:
        print("Failed to connect to Ollama. Please check the service and try again.")
        return None


def query_rag(query_text: str):
    # Prepare the DB.
    embedding_function = get_embedding_function()

    print("Prepare db connection")
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB.
    print("Searching in db...")
    results = db.similarity_search_with_score(query_text, k=10)
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
    query_rag(query_text)
