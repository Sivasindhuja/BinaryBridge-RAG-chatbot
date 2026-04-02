import os
from dotenv import load_dotenv

# Updated imports (latest LangChain structure)
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

# Load environment variables
load_dotenv()


# --- TASK 1: INGESTION ---
def load_documents(directory_path: str):
    loader = DirectoryLoader(
        directory_path,
        glob="**/*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"}   # ✅ fixes encoding error
    )
    documents = loader.load()
    return documents


# --- TASK 2: CHUNKING ---
def chunk_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    return text_splitter.split_documents(documents)


# --- TASK 3: VECTOR DATABASE ---
def setup_vectorstore(chunks):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001"   # ✅ correct Gemini embedding model
    )

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )

    return vectorstore


# --- PIPELINE INITIALIZATION ---
DOCS_DIR = "Documents"

print("Loading documents...")
raw_docs = load_documents(DOCS_DIR)

print("Chunking documents...")
doc_chunks = chunk_documents(raw_docs)

print("Creating vector store...")
vectorstore = setup_vectorstore(doc_chunks)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})


# --- TASK 4: RETRIEVAL & GENERATION ---
def ask_question(question: str):

    llm = ChatGoogleGenerativeAI(
    model="models/gemini-2.5-flash",   # ✅ PERFECT match from your list
    temperature=0.3
    )

    # ✅ FIXED HERE
    docs = retriever.invoke(question)

    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = ChatPromptTemplate.from_template(
        """Answer the question based only on the context below.

Context:
{context}

Question:
{question}
"""
    )

    chain = (
        {"context": lambda x: context, "question": RunnablePassthrough()}
        | prompt
        | llm
    )

    response = chain.invoke(question)

    return response.content, docs


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