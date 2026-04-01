import os
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq

# Load environment variables (API Keys)
load_dotenv()

# --- TASK 1: INGESTION ---
def load_documents(directory_path: str):
    """
    Reads all markdown files from the specified directory.
    """
    loader = DirectoryLoader(
        directory_path,
        glob="**/*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"}
    )
    documents = loader.load()
    return documents


# --- TASK 2: CHUNKING ---
def chunk_documents(documents):
    """
    Splits the loaded documents into smaller, manageable chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    return text_splitter.split_documents(documents)


# --- TASK 3: VECTOR DATABASE ---
def setup_vectorstore(chunks):
    """
    Embeds the document chunks and stores them in a vector database.
    """
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectorstore = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings
    )
    return vectorstore


# --- PIPELINE INITIALIZATION ---
DOCS_DIR = "Documents"

raw_docs = load_documents(DOCS_DIR)
doc_chunks = chunk_documents(raw_docs)
vectorstore = setup_vectorstore(doc_chunks)


# --- TASK 4: RETRIEVAL & GENERATION ---
def ask_question(question: str):
    """
    Main RAG pipeline
    Must return (answer, docs)
    """

    # Retrieve documents
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    docs = retriever.invoke(question) or []

    # Combine context
    # retrieved_context = "\n\n".join([doc.page_content for doc in docs])
    retrieved_context = "\n\n".join([str(doc.page_content) for doc in docs])

    # Prompt
    prompt = (
        "Context:\n"
        f"{retrieved_context}\n\n"
        "Question:\n"
        f"{question}\n\n"
        "Answer based only on context."
    )

    # LLM
    llm = ChatGroq(model="llama-3.1-8b-instant")
    response = llm.invoke(prompt)
    # answer = response.content
    answer = (response.content or "").strip()
    answer = answer.replace("\n", " ")

    return answer, docs


# --- OPTIONAL: CHAT INTERFACE ---
if __name__ == "__main__":
    print("Welcome to the Binary Bridge RAG System!")
    while True:
        user_input = input(
            "\nAsk a question about PMKVY (or type 'exit' to quit): "
        )

        if user_input.lower() in ['exit', 'quit']:
            break

        response, sources = ask_question(user_input)

        print(f"\nAnswer: {response}")
        print("\nSources used:")

        for doc in sources:
            print(
                f"- {doc.metadata.get('source', 'Unknown source')}"
            )