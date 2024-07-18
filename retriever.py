from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import SKLearnVectorStore
from langchain_openai import OpenAIEmbeddings

urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]


def create_retriever_from_urls(urls):
    # Load documents from the URLs
    docs = [WebBaseLoader(url).load() for url in urls]
    docs_list = [item for sublist in docs for item in sublist]

    # Initialize a text splitter with specified chunk size and overlap
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=250, chunk_overlap=0)

    # Split the documents into chunks
    doc_splits = text_splitter.split_documents(docs_list)

    # Add the document chunks to the "vector store" using OpenAIEmbeddings
    vectorstore = SKLearnVectorStore.from_documents(
        documents=doc_splits,
        embedding=OpenAIEmbeddings(model="text-embedding-3-small"),
    )
    return vectorstore.as_retriever(k=4)


retriever = create_retriever_from_urls(urls)
