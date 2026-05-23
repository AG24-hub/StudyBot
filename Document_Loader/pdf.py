from langchain_community.document_loaders import PyPDFLoader
'from langchain_text_splitters import RecursiveCharacterTextSplitter'


data = PyPDFLoader("deeplearning.pdf")

docs = data.load()

print(docs)

'''splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap=10
)

chunks = splitter.split_documents(docs)

print(chunks[0].page_content)'''

''' [\n\n, \n, " ", ""] '''