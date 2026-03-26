import os
from dotenv import load_dotenv
# Hint: Import your necessary LangChain modules here (Text splitters, embeddings, vectorstores, LLMs)

# Load environment variables (API Keys)
load_dotenv()

# --- TASK 1: INGESTION ---
def load_documents(directory_path: str):
    """
    Reads all markdown files from the specified directory.
    
    Args:
        directory_path (str): Path to the folder containing .md files (e.g., "Documents/")
        
    Returns:
        List[Document]: A list of loaded LangChain Document objects.
    """
    # TODO: Implement document loading logic here
    pass


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
    # TODO: Implement your text splitting logic here
    pass


# --- TASK 3: VECTOR DATABASE ---
def setup_vectorstore(chunks):
    """
    Embeds the document chunks and stores them in a vector database.
    
    Args:
        chunks (List[Document]): The chunked documents.
        
    Returns:
        VectorStore: An initialized vector store (e.g., Chroma, FAISS) acting as your retriever.
    """
    # TODO: Implement embedding and vector store initialization here
    pass


# --- PIPELINE INITIALIZATION ---
# We initialize the system once when the script loads so it doesn't re-ingest 
# the documents every time a new question is asked.
DOCS_DIR = "Documents"

# Uncomment these lines once you have implemented the functions above!
# raw_docs = load_documents(DOCS_DIR)
# doc_chunks = chunk_documents(raw_docs)
# vectorstore = setup_vectorstore(doc_chunks)


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
    # TODO: 1. Use the vectorstore to retrieve relevant documents based on the question.
    # TODO: 2. Pass the retrieved documents and the question to an LLM to generate an answer.
    
    answer = "This is a placeholder answer. Implement your LLM generation here."
    docs = [] # Replace with your actual retrieved Document objects
    
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