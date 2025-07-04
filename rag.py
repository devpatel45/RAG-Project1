from uuid import uuid4
from dotenv import load_dotenv
from pathlib import Path
from prompt import PROMPT, EXAMPLE_PROMPT
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains.qa_with_sources.loading import load_qa_with_sources_chain
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

import os

load_dotenv()

# Constants
CHUNK_SIZE = 1000
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
VECTORSTORE_DIR = Path(__file__).parent / "resources/vectorstore"
VECTORSTORE_FILE = VECTORSTORE_DIR / "faiss_index"

llm = None
vector_store = None


def initialize_components():
    """
    Initializes the LLM and loads FAISS vector store if it exists.
    """
    global llm, vector_store

    if llm is None:
        llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.9, max_tokens=500)

    if vector_store is None:
        ef = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"trust_remote_code": True}
        )

        if VECTORSTORE_FILE.exists():
            vector_store = FAISS.load_local(str(VECTORSTORE_FILE), ef)
        else:
            vector_store = None  # Will create in process_urls after loading docs


def process_urls(urls):
    """
    Loads content from URLs, splits into chunks, creates embeddings, and saves to FAISS.
    :param urls: List of webpage URLs
    :return: Yields status messages
    """
    global vector_store

    yield "Initializing Components...✅"
    initialize_components()

    yield "Loading data from URLs...✅"
    loader = UnstructuredURLLoader(urls=urls)
    data = loader.load()

    yield "Splitting text into chunks...✅"
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", " "],
        chunk_size=CHUNK_SIZE
    )
    docs = text_splitter.split_documents(data)

    yield "Creating FAISS vector store...✅"
    ef = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"trust_remote_code": True}
    )
    vector_store = FAISS.from_documents(docs, ef)

    yield "Saving FAISS index to disk...✅"
    VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
    vector_store.save_local(str(VECTORSTORE_FILE))

    yield "Done processing URLs and saving vector store...✅"


def generate_answer(query):
    if not vector_store:
        raise RuntimeError("Vector database is not initialized.")

    qa_chain = load_qa_with_sources_chain(
        llm,
        chain_type="stuff",
        prompt=PROMPT,
        document_prompt=EXAMPLE_PROMPT
    )

    chain = RetrievalQAWithSourcesChain(
        combine_documents_chain=qa_chain,
        retriever=vector_store.as_retriever(),
        reduce_k_below_max_tokens=True,
        max_tokens_limit=8000,
        return_source_documents=True
    )

    result = chain.invoke({"question": query}, return_only_outputs=True)
    sources_docs = [doc.metadata['source'] for doc in result['source_documents']]

    return result['answer'], sources_docs



if __name__ == "__main__":
    # urls = [
    #     "https://www.cnbc.com/2024/12/21/how-the-federal-reserves-rate-policy-affects-mortgages.html",
    #     "https://www.cnbc.com/2024/12/20/why-mortgage-rates-jumped-despite-fed-interest-rate-cut.html"
    # ]

    # for status in process_urls(urls):
    #     print(status)
    # answer, sources = generate_answer("Tell me what was the 30 year fixed mortagate rate along with the date?")
    # print(f"Answer: {answer}")
    # print(f"Sources: {sources}")
    pass
