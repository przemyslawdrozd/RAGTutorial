from langchain.document_loaders import PyPDFDirectoryLoader
DATA_PATH = "data"


def load_documents():
    print("Create loader")
    document_loader = PyPDFDirectoryLoader(DATA_PATH)
    print("Return loaded data")
    return document_loader.load()


print("Call function")
res = load_documents()

print(res)
