import os
from pathlib import Path
from dotenv import load_dotenv

# LangChain imports
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables (API Keys)
load_dotenv()

# Initialize OpenAI API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in .env file")

# Global variables for the pipeline
vectorstore = None
retriever = None
llm = None


# --- TASK 1: INGESTION ---
def load_documents(directory_path: str):
    """
    Reads all markdown files from the specified directory.
    
    Args:
        directory_path (str): Path to the folder containing .md files (e.g., "Documents/")
        
    Returns:
        List[Document]: A list of loaded LangChain Document objects.
    """
    documents = []
    docs_path = Path(directory_path)
    
    # Load all .md files from the directory
    for file_path in docs_path.glob("*.md"):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                # Create a LangChain Document with metadata
                doc = Document(
                    page_content=content,
                    metadata={"source": file_path.name}
                )
                documents.append(doc)
                print(f"✓ Loaded: {file_path.name}")
        except Exception as e:
            print(f"✗ Error loading {file_path.name}: {e}")
    
    print(f"\nTotal documents loaded: {len(documents)}")
    return documents


# --- TASK 2: CHUNKING ---
def chunk_documents(documents):
    """
    Splits the loaded documents into smaller, manageable chunks.
    Uses RecursiveCharacterTextSplitter with optimized parameters for PMKVY documents.
    
    Args:
        documents (List[Document]): The list of loaded documents.
        
    Returns:
        List[Document]: A list of chunked Document objects.
    """
    # Initialize text splitter with optimized parameters
    # chunk_size=1000: Reasonable size for retrieval
    # chunk_overlap=200: Overlap to preserve context at chunk boundaries
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
    
    chunked_docs = []
    for doc in documents:
        chunks = text_splitter.split_documents([doc])
        chunked_docs.extend(chunks)
    
    print(f"Total chunks created: {len(chunked_docs)}")
    return chunked_docs


# --- TASK 3: VECTOR DATABASE ---
def setup_vectorstore(chunks):
    """
    Embeds the document chunks and stores them in a vector database (ChromaDB).
    
    Args:
        chunks (List[Document]): The chunked documents.
        
    Returns:
        VectorStore: A Chroma vector store for retrieval.
    """
    # Initialize OpenAI embeddings
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=OPENAI_API_KEY
    )
    
    # Create or load ChromaDB collection
    # persist_directory ensures data is saved locally
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="pmkvy_documents",
        persist_directory="./chroma_db"
    )
    
    print(f"✓ Vector store created with {len(chunks)} chunks")
    return vectorstore


# --- HELPER: CONTEXT FORMATTING ---
def format_context(docs, max_docs=5):
    """
    Formats retrieved documents into a context string for the LLM.
    
    Args:
        docs (List[Document]): Retrieved documents.
        max_docs (int): Maximum number of documents to include.
        
    Returns:
        str: Formatted context string.
    """
    context = ""
    for i, doc in enumerate(docs[:max_docs], 1):
        context += f"\n[Document {i}]\n{doc.page_content}\n"
    return context


# --- PIPELINE INITIALIZATION ---
def initialize_pipeline():
    """
    Initialize the RAG pipeline once on startup.
    Loads documents, chunks them, and creates the vector store.
    """
    global vectorstore, retriever, llm
    
    print("🚀 Initializing RAG Pipeline...")
    
    # Step 1: Load documents
    docs_dir = "Documents"
    raw_docs = load_documents(docs_dir)
    
    # Step 2: Chunk documents
    print("\n📦 Chunking documents...")
    doc_chunks = chunk_documents(raw_docs)
    
    # Step 3: Setup vector store
    print("\n🔍 Setting up vector store...")
    vectorstore = setup_vectorstore(doc_chunks)
    
    # Step 4: Initialize retriever
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}  # Retrieve top 4 similar documents
    )
    
    # Step 5: Initialize LLM
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0.3,  # Lower temperature for more factual answers
        api_key=OPENAI_API_KEY
    )
    
    print("\n✅ RAG Pipeline initialized successfully!")


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
    if vectorstore is None or retriever is None or llm is None:
        raise RuntimeError("Pipeline not initialized. Call initialize_pipeline() first.")
    
    # Step 1: Retrieve relevant documents
    retrieved_docs = retriever.invoke(question)
    
    # Step 2: Format context from retrieved documents
    context = format_context(retrieved_docs)
    
    # Step 3: Create prompt template
    prompt_template = ChatPromptTemplate.from_template("""
You are a helpful assistant answering questions about Indian government PMKVY (Pradhan Mantri Kaushal Vikas Yojana) schemes.

Based on the following documents, answer the user's question accurately and concisely. 
If the information is not in the documents, say "I don't have enough information to answer this question."

Documents:
{context}

Question: {question}

Answer:""")
    
    # Step 4: Run the LLM with context and question
    chain = prompt_template | llm
    response = chain.invoke({
        "context": context,
        "question": question
    })
    
    # Step 5: Extract answer text
    answer = response.content
    
    # Return answer and retrieved documents (required by RAGAS evaluation)
    return answer, retrieved_docs


# --- OPTIONAL: CHAT INTERFACE ---
if __name__ == "__main__":
    # Initialize pipeline once at startup
    initialize_pipeline()
    
    print("\n" + "="*60)
    print("Welcome to the Binary Bridge RAG System!")
    print("="*60)
    
    while True:
        user_input = input("\nAsk a question about PMKVY (or type 'exit' to quit): ")
        if user_input.lower() in ['exit', 'quit']:
            print("Goodbye!")
            break
        
        try:
            # Call the pipeline
            response, sources = ask_question(user_input)
            print(f"\n📝 Answer:\n{response}")
            print(f"\n📚 Sources ({len(sources)} documents):")
            for i, doc in enumerate(sources, 1):
                print(f"   {i}. {doc.metadata.get('source', 'Unknown')}")
        except Exception as e:
            print(f"\n❌ Error: {e}")
        
        print(f"\nAnswer: {response}")
        print("\nSources used:")
        for doc in sources:
             print(f"- {doc.metadata.get('source', 'Unknown source')}")