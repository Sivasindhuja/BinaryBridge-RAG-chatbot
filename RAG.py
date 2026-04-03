import os
from dotenv import load_dotenv

# LangChain imports
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import TextLoader,DirectoryLoader

load_dotenv()


# --- TASK 1: INGESTION ---
def load_documents(directory_path: str):

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

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=150
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

retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 8, "fetch_k": 15}
)

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash"
)
# llm = ChatGoogleGenerativeAI(
#     model="models/gemini-2.5-flash"
# )

# --- TASK 4: RETRIEVAL & GENERATION ---
def ask_question(question: str):

    docs = retriever.invoke(question)

    seen = set()
    clean_texts = []

    for doc in docs:
        text = doc.page_content.strip()
    
        if text not in seen:
            seen.add(text)
            clean_texts.append(text)

    context = "\n\n".join(clean_texts)
    prompt = f"""
    You are an AI assistant.

    Answer the question clearly and completely using ONLY the context below.
    - Think carefully before answering
    - Do not guess
    - If multiple topics exist, choose the most relevant one
    - Give a complete sentence answer with 3-4 lines

    If the answer is not present, say "Not found in context".


    Context:
    {context}

    Question:
    {question}
    """

    response = llm.invoke(prompt)
    
    import re  # you can also move this to top of file

    answer = response.content

    # 🔹 Remove markdown (**bold**)
    answer = re.sub(r'\*\*(.*?)\*\*', r'\1', answer)

    # 🔹 Remove single *
    answer = answer.replace("*", "")
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

        unique_sources = list(set([doc.metadata.get("source") for doc in sources]))

        for src in unique_sources:
            print(f"- {src}")