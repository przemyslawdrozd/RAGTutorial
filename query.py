import argparse
from langchain_ollama import OllamaEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
from collections import defaultdict
import sqlite3
from query_rag_utils import (
    CHROMA_PATH,
    load_config,
    load_embedding_model,
    load_llm_model,
    filter_results_by_score,
)
from config_loader import (
    SQL_PROMPT_TEMPLATE,
    PDF_PROMPT_TEMPLATE,
)

CHROMA_PATH = "chroma"
SQLITE_DB_PATH = "chroma/sqlitedb"
EMBEDDING_MODEL = load_config().get("embedding_model", "mistral")
SIMILARITY_THRESHOLD = load_config().get("similarity_threshold", 0.85)
MULTIPLE_RESPONSES = load_config().get("multiple_responses", False)


def get_table_columns(db_path, table_name):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(f"PRAGMA table_info({table_name})")
    columns = cursor.fetchall()
    conn.close()
    return columns


def get_embedding_function():
    try:
        return OllamaEmbeddings(model=EMBEDDING_MODEL, temperature=0)
    except ConnectionError:
        print("Failed to connect to Ollama. Please check the service and try again.")
        return None


def query_rag(query_text: str, k=15):
    global MULTIPLE_RESPONSES

    print("Loading embedding model...")
    embedding_function = load_embedding_model()

    # 🔍 Lets decrease k for SQL
    if "sql" in query_text.lower():
        print("🛠️ SQL detected in query — reducing k for schema search.")
        k = 5  # TODO Look for optimal match to get only schema data
        MULTIPLE_RESPONSES = False

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

    if MULTIPLE_RESPONSES:
        # Grouped per document
        grouped = defaultdict(list)
        for doc, _ in filtered:
            source = doc.metadata.get("source", "unknown")
            grouped[source].append(doc.page_content)

        final_responses = {}
        for source, pages in grouped.items():
            print("Querying for source:", source)

            # 🚫 Skip SQL schema sources
            if source.startswith("sqlitedb:"):
                print(f"Skipping SQL schema source: {source}")
                continue

            # 🚀 Remove duplicated chunks (based on content only)
            deduped_pages = list(dict.fromkeys(pages))  # preserves order

            context_text = "\n\n---\n\n".join(deduped_pages)

            print("⚙️ Using PDF/general prompt")
            prompt_template = ChatPromptTemplate.from_template(PDF_PROMPT_TEMPLATE)

            print("Context:", context_text)
            prompt = prompt_template.format(context=context_text, question=query_text)
            response = model.invoke(prompt)
            print("Got response for source", source)
            final_responses[source] = response

        for source, response in final_responses.items():
            print(f"\n=== Answer from {source} ===\n{response}\n")

        return final_responses

    else:
        # Look for SQL Schema
        schema_docs = [
            doc for doc, _ in filtered
            if doc.metadata.get("schema")
        ]

        if schema_docs:
            print("🔍 Found schema SQL schema — switching to SQL generation.", schema_docs)

            # 🚀 Retrieve text schema
            schema_text = "\n\n".join([doc.page_content for doc in schema_docs])

            # 🚀 Get found table name
            table_source = schema_docs[0].metadata.get("source")  # np. 'sqlitedb:Guests'
            table_name = table_source.split(":")[1]  # 'Guests'
            print("Using table:", table_name)

            # 🚀 Fetch all columns
            columns = get_table_columns(SQLITE_DB_PATH, table_name)
            all_column_names = ', '.join([f'"{col[1]}"' for col in columns])

            # 🚀 Generate SQL
            prompt_template = ChatPromptTemplate.from_template(SQL_PROMPT_TEMPLATE)
            prompt = prompt_template.format(schema=schema_text, question=query_text, column_list=all_column_names)
            sql_query = model.invoke(prompt)
            print("Generated SQL query:\n", sql_query)

            # 🚀 Run SQL on SQLite
            conn = sqlite3.connect(SQLITE_DB_PATH)
            cursor = conn.cursor()
            try:
                cursor.execute(sql_query)
                results = cursor.fetchall()
                print("SQL Query Results:\n", results)
                return results
            except Exception as e:
                print(f"❌ SQL execution error: {e}")
                return None
            finally:
                conn.close()

        else:
            print("📝 No schema found — proceeding with standard RAG flow.")

            # 🚀 Standard PDF RAG flow
            context_text = "\n\n---\n\n".join([doc.page_content for doc, _ in filtered])
            prompt_template = ChatPromptTemplate.from_template(PDF_PROMPT_TEMPLATE)
            prompt = prompt_template.format(context=context_text, question=query_text)
            response = model.invoke(prompt)

            sources = [
                doc.metadata.get("id")
                for doc, _ in filtered
                if not doc.metadata.get("schema")  # Skip SQL Schemas
            ]

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
