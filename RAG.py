import os
from dotenv import load_dotenv

# LangChain imports
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()


# --- TASK 1: INGESTION ---
def load_documents(directory_path: str):

    loader = DirectoryLoader(directory_path, glob="**/*.md")

    documents = loader.load()

    return documents


# --- TASK 2: CHUNKING ---
def chunk_documents(documents):

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = text_splitter.split_documents(documents)

    return chunks


# --- TASK 3: VECTOR DATABASE ---
def setup_vectorstore(chunks):

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )

    return vectorstore


# --- PIPELINE INITIALIZATION ---
DOCS_DIR = "Documents"

raw_docs = load_documents(DOCS_DIR)
doc_chunks = chunk_documents(raw_docs)
vectorstore = setup_vectorstore(doc_chunks)

retriever = vectorstore.as_retriever(search_kwargs={"k":3})

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash"
)
# llm = ChatGoogleGenerativeAI(
#     model="models/gemini-2.5-flash"
# )

# --- TASK 4: RETRIEVAL & GENERATION ---
def ask_question(question: str):

    docs = retriever.invoke(question)

    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = f"""
    Use the following context to answer the question.

    Context:
    {context}

    Question:
    {question}
    """

    response = llm.invoke(prompt)

    answer = response.content

    return answer, docs


# --- OPTIONAL: CHAT INTERFACE ---
if __name__ == "__main__":
    print("Welcome to the Binary Bridge RAG System!")

    while True:
        user_input = input("\nAsk a question about PMKVY (or type 'exit' to quit): ")

        if user_input.lower() in ['exit', 'quit']:
            break

        response, sources = ask_question(user_input)

        print(f"\nAnswer: {response}")

        print("\nSources used:")

        for doc in sources:
            print(f"- {doc.metadata.get('source', 'Unknown source')}")