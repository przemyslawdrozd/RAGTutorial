from langchain_community.document_loaders import PyPDFDirectoryLoader, CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from config_loader import load_config
import pandas as pd
import os

DATA_PATH = "data"
CHROMA_PATH = "chroma"

MODEL_NAME = load_config().get("embedding_model", "mistral")


def load_documents() -> list[Document]:
    print("Loading documents from:", DATA_PATH)
    documents = []

    # Load PDFs
    print("→ Scanning for PDFs...")
    pdf_loader = PyPDFDirectoryLoader(DATA_PATH)
    documents.extend(pdf_loader.load())

    # Load CSVs with custom loader
    print("→ Scanning for CSVs...")
    for file in os.listdir(DATA_PATH):
        if file.endswith(".csv"):
            file_path = os.path.join(DATA_PATH, file)
            print(f"→ Loading CSV: {file}")
            try:
                docs_from_csv = generic_csv_loader(file_path)

                # Generate summary
                summary_doc = summarize_csv(file_path)
                documents.extend([summary_doc] + docs_from_csv)
                print(f"   ✅ Loaded {len(docs_from_csv)} rows + 1 summary from {file}")

                documents.extend(docs_from_csv)
                print(f"   ✅ Loaded {len(docs_from_csv)} rows from {file}")
            except Exception as e:
                print(f"   ❌ Failed to load {file}: {e}")

    print(f"✅ Total loaded documents: {len(documents)}")
    return documents


def row_to_string(row: dict) -> str:
    return "; ".join([f"{key}: {value}" for key, value in row.items()])


def summarize_csv(file_path: str) -> Document:
    df = pd.read_csv(file_path)
    columns = list(df.columns)
    num_rows = len(df)

    summary_lines = [
        f"The dataset '{os.path.basename(file_path)}' contains {num_rows} rows.",
        f"It includes the following columns: {', '.join(columns)}.",
    ]

    # Try summarizing a few useful columns
    for col in columns:
        if df[col].nunique() < 50 and df[col].dtype == object:
            unique_values = df[col].dropna().unique()
            summary_lines.append(f"Column '{col}' has {len(unique_values)} unique values: {', '.join(map(str, unique_values[:10]))}...")

    summary = "\n".join(summary_lines)

    return Document(
        page_content=summary,
        metadata={"source": file_path, "summary": True}
    )

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
    # This will create IDs like "data/monopoly.pdf:6:2"
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
