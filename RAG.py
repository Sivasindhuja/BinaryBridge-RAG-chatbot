import os
from dotenv import load_dotenv

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("❌ GOOGLE_API_KEY or GEMINI_API_KEY must be set in your .env file!")

DOCS_DIR = "Documents"

# --- Load documents ---
def load_documents(directory_path):
    loader = DirectoryLoader(
        directory_path,
        glob="**/*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"}
    )
    return loader.load()

# --- Chunk documents ---
def chunk_documents(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    return splitter.split_documents(documents)

# --- Vector store ---
def setup_vectorstore(chunks):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        api_key=GOOGLE_API_KEY
    )
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    return vectorstore

# --- Main ---
raw_docs = load_documents(DOCS_DIR)
doc_chunks = chunk_documents(raw_docs)
vectorstore = setup_vectorstore(doc_chunks)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

def ask_question(question: str):
    llm = ChatGoogleGenerativeAI(
        model="models/gemini-2.5-flash",
        temperature=0.3,
        api_key=GOOGLE_API_KEY
    )

    # ✅ Correct way to get documents
    docs = retriever.get_relevant_documents(question)
    context = "\n\n".join([doc.page_content for doc in docs]) if docs else "No context"

    prompt = ChatPromptTemplate.from_template(
        """Answer the question based only on the context below.

Context:
{context}

Question:
{question}"""
    )

    chain = (
        {"context": lambda x: context, "question": RunnablePassthrough()}
        | prompt
        | llm
    )

    response = chain.invoke(question)
    return response.content, docs

if __name__ == "__main__":
    print("🤖 Binary Bridge RAG Ready!")
    while True:
        user_input = input("\nAsk a question (or 'exit'): ")
        if user_input.lower() in ["exit", "quit"]:
            break
        answer, sources = ask_question(user_input)
        print(f"\n📝 Answer: {answer}")
        print("📚 Sources used:")
        for doc in sources:
            print(f"- {doc.metadata.get('source', 'Unknown')}")