import os
from dotenv import load_dotenv

# -------------------------------
# LangChain & VectorStore imports
# -------------------------------
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders.markdown import UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
#from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
#from sentence_transformers import SentenceTransformer

# Load environment variables
load_dotenv()

# -------------------------------
# Custom Hugging Face embedding wrapper
# -------------------------------
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import Chroma

class HFEmbedding:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts):
        # convert NumPy array to list of lists
        return self.model.encode(texts, show_progress_bar=True).tolist()

    def embed_query(self, text):
        # return single list for a query
        return self.model.encode([text], show_progress_bar=False)[0].tolist()

# -------------------------------
# TASK 1: LOAD DOCUMENTS
# -------------------------------
def load_documents(directory_path: str):
    loader = DirectoryLoader(
        directory_path,
        glob="*.md",
        loader_cls=UnstructuredMarkdownLoader,
        show_progress=True
    )
    return loader.load()

# -------------------------------
# TASK 2: CHUNK DOCUMENTS
# -------------------------------
def chunk_documents(documents, chunk_size=500, chunk_overlap=100):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(documents)

# -------------------------------
# TASK 3: VECTORSTORE SETUP
# -------------------------------
def setup_vectorstore(chunks):
    embedding = HFEmbedding()

    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embedding,
        persist_directory="chroma_db"
    )
    vectordb.persist()
    return vectordb

# -------------------------------
# TASK 4: RAG PIPELINE
# -------------------------------
def ask_question(question: str):
    try:
        # Step 1: Retrieve documents using similarity_search (works across versions)
        retrieved_docs = vectorstore.similarity_search(question, k=4)

        # Step 2: Build context
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])

        # Step 3: Prompt
        prompt = (
            "You are an expert on Indian government schemes.\n"
            "Answer ONLY from the provided context.\n"
            "If the answer is not in the context, say: 'Not available in context'.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\nAnswer:"
        )

        # Step 4: LLM
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0.2
        )

        # Step 5: Generate answer
        response = llm.invoke(prompt)
        answer = response.content if hasattr(response, "content") else str(response)

        return answer, retrieved_docs

    except Exception as e:
        print(f"❌ Error: {e}")
        return "[Error generating answer]", []

# -------------------------------
# INITIALIZATION
# -------------------------------
DOCS_DIR = "Documents"

print("Loading documents...")
raw_docs = load_documents(DOCS_DIR)

print("Chunking documents...")
doc_chunks = chunk_documents(raw_docs)

print("Creating vector database...")
vectorstore = setup_vectorstore(doc_chunks)

print("✅ RAG system ready!")

# -------------------------------
# CHAT INTERFACE
# -------------------------------
if __name__ == "__main__":
    print("\n💬 Welcome to the Binary Bridge RAG System!")

    while True:
        user_input = input("\nAsk a question (type 'exit' to quit): ")

        if user_input.lower() in ["exit", "quit"]:
            print("👋 Exiting...")
            break

        answer, sources = ask_question(user_input)

        print(f"\n🧠 Answer:\n{answer}")

        print("\n📄 Sources used:")
        for doc in sources:
            print(f"- {doc.metadata.get('source', 'Unknown source')}")