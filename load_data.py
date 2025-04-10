from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

DATA_PATH = "data"
CHROMA_PATH = "chroma"


def load_documents():
    print("Create loader")
    document_loader = PyPDFDirectoryLoader(DATA_PATH)
    print("Return loaded data")
    return document_loader.load()


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
        model = "mistral"
        return OllamaEmbeddings(model=model)
    except ConnectionError:
        print("Failed to connect to Ollama. Please check the service and try again.")
        return None


def calculate_chunk_ids(chunks):
    # This will create IDs like "data/monopoly.pdf:6:2"
    # Page Source : Page Number : Chunk Index

    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        # If the page ID is the same as the last one, increment the index.
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        # Calculate the chunk ID.
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        # Add it to the page meta-data.
        chunk.metadata["id"] = chunk_id

    return chunks


def add_to_chroma(doc_chunks: list[Document], ollama_model):
    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=ollama_model
    )
    # Calculate Page IDs.
    chunks_with_ids = calculate_chunk_ids(doc_chunks)

    # Retrieve existing document IDs to prevent duplicates.
    existing_items = db.get(include=[])
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")

    # Process each document one by one.
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            try:
                # Add a single document.
                db.add_documents([chunk], ids=[chunk.metadata["id"]])
                print(f"Added document ID: {chunk.metadata['id']}")
            except Exception as e:
                print(f"Failed to add document ID {chunk.metadata['id']}: {e}")
        else:
            print(f"Document ID {chunk.metadata['id']} already exists in DB.")


if __name__ == "__main__":
    print("Call function")
    loaded_documents = load_documents()

    print("Create Document chunks")
    chunks = split_documents(loaded_documents)

    ollama_model = get_embedding_function()
    print("ollama model", ollama_model)

    print("Load data into Chroma db")
    add_to_chroma(chunks, ollama_model)
    print("Finished")



