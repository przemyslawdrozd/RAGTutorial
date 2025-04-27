from langchain_community.document_loaders import PyPDFDirectoryLoader, CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from config_loader import load_config
import sqlite3
import pandas as pd
import os

DATA_PATH = "data"
CHROMA_PATH = "chroma"
SQLITE_DB_PATH = "chroma/sqlitedb"
MODEL_NAME = load_config().get("embedding_model", "mistral")
sql_column_summary = load_config().get("sql_column_summary")


def load_documents() -> list[Document]:
    print("Loading documents from:", DATA_PATH)
    documents = []

    # Load PDFs
    print("→ Scanning for PDFs...")
    pdf_loader = PyPDFDirectoryLoader(DATA_PATH)
    documents.extend(pdf_loader.load())

    # Load CSVs to SQLite + generate schema docs
    print("→ Scanning for CSVs...")
    for file in os.listdir(DATA_PATH):
        if file.endswith(".csv"):
            file_path = os.path.join(DATA_PATH, file)
            print(f"→ Loading CSV into SQLite: {file}")
            try:
                table_name = load_csv_to_sqlite(file_path, SQLITE_DB_PATH)
                schema_doc = generate_schema_doc(SQLITE_DB_PATH, table_name)
                documents.append(schema_doc)
                print(f"   ✅ Loaded {file} into SQLite as table {table_name}")
            except Exception as e:
                print(f"   ❌ Failed to load {file}: {e}")

    print(f"✅ Total loaded documents: {len(documents)}")
    return documents


def load_csv_to_sqlite(file_path, db_path):
    df = pd.read_csv(file_path)

    # 🚀 Snake_case for column name
    df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]

    # 🚀 Strip spaces
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].apply(lambda x: x.strip() if isinstance(x, str) else x)

    # 🚀 Save to SQLite
    conn = sqlite3.connect(db_path)
    table_name = os.path.splitext(os.path.basename(file_path))[0].lower().replace(" ", "_")  # też snake_case dla tabeli

    df.to_sql(table_name, conn, if_exists="replace", index=False)
    conn.close()

    print(f"✅ Loaded CSV '{file_path}' into table '{table_name}' with cleaned columns and data.")
    return table_name


def generate_schema_doc(db_path, table_name):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute(f"PRAGMA table_info({table_name})")
    columns = cursor.fetchall()

    column_descriptions = []
    for col in columns:
        col_name_original = col[1]
        col_name_snake = col_name_original.strip().lower().replace("\"", "").replace(" ", "_")
        col_type = col[2]

        # 🚀 RENAME COLUMN - if different name
        if col_name_original != col_name_snake:
            print(f"Renaming column: {col_name_original} -> {col_name_snake}")
            cursor.execute(f'ALTER TABLE `{table_name}` RENAME COLUMN `{col_name_original}` TO `{col_name_snake}`')

        cursor.execute(
            f"SELECT DISTINCT `{col_name_snake}` FROM `{table_name}` WHERE `{col_name_snake}` IS NOT NULL LIMIT 5;")
        raw_values = cursor.fetchall()

        # Set limit to 20 chars
        unique_values = []
        for row in raw_values:
            val = str(row[0])
            if len(val) > 20:
                val = val[:17] + "..."
            unique_values.append(val)

        value_info = f" — example values: {', '.join(unique_values)}" if unique_values else ""
        column_descriptions.append(f"- {col_name_snake} ({col_type}){value_info}")

    conn.close()

    schema_text = (
            f"Table: {table_name}\n"
            f"Columns:\n" + "\n".join(column_descriptions)
    )

    all_column_names = [f'"{col[1]}"' for col in columns]

    schema_text += (
            f"\n\n{sql_column_summary}\n" + ", ".join(
        all_column_names)
    )

    print("Created schema_text", schema_text)
    return Document(page_content=schema_text, metadata={"source": f"{db_path}:{table_name}", "schema": True})


def row_to_string(row: dict) -> str:
    return "; ".join([f"{key}: {value}" for key, value in row.items()])


def generic_csv_loader(file_path):
    df = pd.read_csv(file_path)
    docs = []

    for _, row in df.iterrows():
        text = row_to_string(row.to_dict())
        docs.append(Document(page_content=text, metadata={"source": file_path}))

    return docs


def split_documents(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)


def get_embedding_function():
    try:
        return OllamaEmbeddings(model=MODEL_NAME)
    except ConnectionError:
        print("Failed to connect to Ollama. Please check the service and try again.")
        return None


def calculate_chunk_ids(chunks):
    # This will create IDs like "data/pdf:6:2"
    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")

        # Default to 0 if no page (e.g. CSV)
        page = chunk.metadata.get("page", 0)

        current_page_id = f"{source}:{page}"

        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id
        chunk.metadata["id"] = chunk_id

    return chunks


def add_to_chroma(doc_chunks: list[Document], ollama_model):
    db = Chroma(
        persist_directory=f"{CHROMA_PATH}/{MODEL_NAME}",
        embedding_function=ollama_model
    )

    chunks_with_ids = calculate_chunk_ids(doc_chunks)

    existing_items = db.get(include=[])
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")

    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            try:
                db.add_documents([chunk], ids=[chunk.metadata["id"]])
                print(f"Added document ID: {chunk.metadata['id']}")
            except Exception as e:
                print(f"❌ Failed to add document ID {chunk.metadata['id']}: {e}")
        else:
            print(f"✅ Document ID {chunk.metadata['id']} already exists in DB.")


if __name__ == "__main__":
    print("Call function")
    loaded_documents = load_documents()

    print("Create Document chunks")
    chunks = split_documents(loaded_documents)

    ollama_model = get_embedding_function()
    print("Embedding model:", ollama_model)

    print("Load data into Chroma db")
    add_to_chroma(chunks, ollama_model)
    print("✅ Finished")
