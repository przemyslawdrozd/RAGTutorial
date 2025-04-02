from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document

DATA_PATH = "data"


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


print("Call function")
loaded_documents = load_documents()
chunks = split_documents(loaded_documents)

print(chunks[0])
print(len(chunks))
