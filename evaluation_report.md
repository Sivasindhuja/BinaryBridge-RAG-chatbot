# RAG Evaluation Report

## Performance Summary
The Retrieval-Augmented Generation (RAG) system demonstrates strong performance across answering queries related to PMKVY schemes. The integration of LangChain's retrieval pipeline with Google's Gemini-Pro model provided accurate, contextually relevant answers for most test questions based on the provided documents. The structured vector search implementation efficiently filtered down irrelevant text and fed the LLM with focused context.

## Strengths
- **Accuracy**: It adheres well to the system prompt's instruction to only provide answers found in the context, preventing hallucinations.
- **Context Retrieval Precision**: The use of HuggingFace's `all-MiniLM-L6-v2` embeddings coupled with ChromaDB provides excellent dense vector matching for queries.
- **Robustness**: The chatbot is well-handled in a Streamlit interface that caches the RAG components to prevent latency on repetitive document ingestions.

## Weaknesses
- **Cross-document queries**: Finding context that spans across `PMKVY_RPL.md` and `PMKVY_STT_Scheme.md` sometimes challenges the top-K=5 retrieval limit if the chunks are highly dispersed.
- **Slow LLM Latency**: External API calls to generative models can occasionally introduce delay in the interactive workflow.

## Chunking Impact
- **Strategy**: A `RecursiveCharacterTextSplitter` was used with `chunk_size = 500` and `chunk_overlap = 100`.
- **Impact**: This granularity proved highly beneficial because government scheme documents usually feature dense definitions and bullet points. At 500 characters, most bulleted points fit into a single chunk, preserving semantic meaning without muddying the context with unrelated adjacent sections. The 100-character overlap prevents key terms and sentence subjects at the boundary from getting cut off.

## Suggestions for Improvement
1. **Hybrid Search**: Combine lexical (BM25) and dense embeddings to improve keyword-specific searches alongside semantic similarity.
2. **Metadata Filtering**: If schemes are categorized, tagging chunks with metadata like "Scheme Type" would allow precision filtering before vector search.
3. **Advanced RAG (Parent Document Retriever)**: Store small chunks for precise retrieval but pass the parent document to the LLM to give the model broader context without hitting token limits.
