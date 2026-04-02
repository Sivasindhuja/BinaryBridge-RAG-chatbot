import os
import glob
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings

# Load environment variables (API Keys)
load_dotenv()

# LangChain uses OPENAI_API_KEY for ChatOpenAI by default, we'll set base_url
groq_key = os.environ.get("GROQ_API_KEY", "")
if groq_key:
    os.environ["OPENAI_API_KEY"] = groq_key

# --- TASK 1: INGESTION ---
def load_documents(directory_path: str):
    """
    Reads all markdown files from the specified directory.
    
    Args:
        directory_path (str): Path to the folder containing .md files (e.g., "Documents/")
        
    Returns:
        List[Document]: A list of loaded LangChain Document objects.
    """
    docs = []
    files = glob.glob(os.path.join(directory_path, "**", "*.md"), recursive=True)
    for file_path in files:
        try:
            loader = TextLoader(file_path, encoding='utf-8')
            docs.extend(loader.load())
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    return docs


# --- TASK 2: CHUNKING ---
def chunk_documents(documents):
    """
    Splits the loaded documents into smaller, manageable chunks.
    Experiment with different chunk sizes and overlaps!
    
    Args:
        documents (List[Document]): The list of loaded documents.
        
    Returns:
        List[Document]: A list of chunked Document objects.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=150,
        separators=["\n\n", "\n", " ", ""]
    )
    return text_splitter.split_documents(documents)


# --- TASK 3: VECTOR DATABASE ---
def setup_vectorstore(chunks):
    """
    Embeds the document chunks and stores them in a vector database.
    
    Args:
        chunks (List[Document]): The chunked documents.
        
    Returns:
        VectorStore: An initialized vector store (e.g., Chroma, FAISS) acting as your retriever.
    """
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(chunks, embedding=embeddings)
    return vectorstore


# --- PIPELINE INITIALIZATION ---
# We initialize the system once when the script loads so it doesn't re-ingest 
# the documents every time a new question is asked.
DOCS_DIR = "Documents"

# Uncomment these lines once you have implemented the functions above!
raw_docs = load_documents(DOCS_DIR)
doc_chunks = chunk_documents(raw_docs)
vectorstore = setup_vectorstore(doc_chunks)


# --- TASK 4: RETRIEVAL & GENERATION ---
def ask_question(question: str):
    """
    The main RAG pipeline function. Takes a user question, retrieves relevant context, 
    and generates an answer using an LLM.
    
    CRITICAL: This function must return a tuple of (answer, source_documents) for 
    the RAGAS evaluation script to work properly.
    
    Args:
        question (str): The user's question.
        
    Returns:
        tuple: (answer (str), docs (List[Document]))
            - answer: The generated text response.
            - docs: The list of Document objects retrieved from the vector store and used as context.
    """
    # 1. Use the vectorstore to retrieve relevant documents based on the question.
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    docs = retriever.invoke(question)
    
    # 2. Pass the retrieved documents and the question to an LLM to generate an answer.
    llm = ChatOpenAI(
        model="llama-3.3-70b-versatile", 
        temperature=0,
        base_url="https://api.groq.com/openai/v1"
    )
    
    context = "\n\n".join([doc.page_content for doc in docs])
    
    prompt = f"""You are a helpful assistant. Answer the user's question based strictly on the following context.
    Do not use outside information. If the answer is not contained in the context, say so.
    
    Context:
    {context}
    
    Question: {question}
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
            
        # Call the pipeline
        response, sources = ask_question(user_input)
        
        print(f"\nAnswer: {response}")
        print("\nSources used:")
        for doc in sources:
             print(f"- {doc.metadata.get('source', 'Unknown source')}")