import os
import hashlib
import pickle
from typing import List, Tuple, Optional

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import LlamaCpp
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import Document

def get_file_hash(file_path: str) -> str:
    """Generate hash for file to check if embeddings need to be updated"""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def get_vector_store_path(pdf_path: str, model_name: str = "all-MiniLM-L6-v2") -> str:
    """Generate path for vector store files"""
    base_name = os.path.splitext(os.path.basename(pdf_path))[0]
    model_hash = hashlib.md5(model_name.encode()).hexdigest()[:8]
    return f"vector_store/{base_name}_{model_hash}"


def load_and_split_documents(pdf_path: str, chunk_size: int = 400, chunk_overlap: int = 50) -> List[Document]:
    """Load PDF and split into documents"""
    print("Loading PDF document...")
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    
    print("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks")
    
    return chunks

def create_vector_store(chunks: List[Document], model_name: str = "all-MiniLM-L6-v2") -> FAISS:
    """Create FAISS vector store from document chunks"""
    print("Creating embeddings and vector store...")
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore

def save_vector_store(vectorstore: FAISS, pdf_hash: str, vector_store_path: str):
    """Save vector store to disk"""
    os.makedirs("vector_store", exist_ok=True)
    vectorstore.save_local(vector_store_path)
    print(f"Vector store saved to {vector_store_path}")

def process_document_with_vector_store(
    pdf_path: str, 
    model_name: str = "all-MiniLM-L6-v2", 
    chunk_size: int = 400, 
    chunk_overlap: int = 50
) -> FAISS:
    """Process document with vector store caching"""
    vector_store_path = get_vector_store_path(pdf_path, model_name)
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    
    # 檢查檔案是否存在
    faiss_file = f"{vector_store_path}/index.faiss"
    pkl_file = f"{vector_store_path}/index.pkl"
    
    if os.path.exists(faiss_file) and os.path.exists(pkl_file):
        try:
            vectorstore = FAISS.load_local(
                vector_store_path, 
                embeddings, 
                allow_dangerous_deserialization=True
            )
            print("Using existing vector store")
            return vectorstore
        except Exception as e:
            print(f"Error loading vector store: {e}")
            print("Creating new vector store")
    else:
        print("Creating new vector store")
    
    # 建立新的 vector store
    chunks = load_and_split_documents(pdf_path, chunk_size, chunk_overlap)
    vectorstore = create_vector_store(chunks, model_name)
    vectorstore.save_local(vector_store_path)
    return vectorstore

def create_custom_prompt() -> PromptTemplate:
    """Create custom prompt template for musical theory consultation"""
    template = """You are a cheerful and patient musical theory teacher. Explain the answer using clear and beginner-friendly language, and try to be interactive by throwing insightful questions.

Context:
{context}

User: {question}

Please help your student with a detailed and helpful answer, and try to guide the student to ask right questions if he seem astray:"""

    return PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )

def create_llm(model_path: str = "models/llama-2-7b.Q4_K_M.gguf", n_ctx: int = 4096, threads: int = 8, batch_size: int = 512) -> LlamaCpp:
    """Create LLM instance"""
    return LlamaCpp(
        model_path=model_path,
        n_ctx=n_ctx,
        temperature=0.1,
        max_tokens=512,
        top_p=1,
        verbose=True,
        n_gpu_layers=100,
        device="cuda",
    )

def create_qa_chain(vectorstore: FAISS, llm: LlamaCpp) -> RetrievalQA:
    """Create QA chain with custom prompt"""
    prompt = create_custom_prompt()
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True,
    )
    
    return qa_chain

def rag_pipeline(
    llm: LlamaCpp,
    pdf_path: str, 
    query: str,
    chunk_size: int = 400, 
    chunk_overlap: int = 50, 
    k: int = 3, 
    model_name: str = "all-MiniLM-L6-v2",
) -> Tuple[str, List[Document], List[float]]:
    """Complete RAG pipeline using LangChain"""
    print("=== LangChain RAG Pipeline ===")
    
    # Process document with vector store
    vectorstore = process_document_with_vector_store(
        pdf_path, model_name, chunk_size, chunk_overlap
    )
    
        
    # Create QA chain
    print("Creating QA chain...")
    qa_chain = create_qa_chain(vectorstore, llm)
    
    # Get response
    print("Generating response...")
    result = qa_chain({"query": query})
    
    # Extract results
    answer = result["result"]
    source_documents = result["source_documents"]
    
    # Get similarity scores (approximate)
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    query_embedding = embeddings.embed_query(query)
    
    scores = []
    for doc in source_documents:
        doc_embedding = embeddings.embed_query(doc.page_content)
        # Calculate cosine similarity
        import numpy as np
        similarity = np.dot(query_embedding, doc_embedding) / (
            np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
        )
        scores.append(similarity)
    
    return answer, source_documents, scores

# Example usage
if __name__ == "__main__":
    pdf_path = "data/Sample.pdf"
    query = "樂理的基礎知識有哪些？"

    print("=== LangChain FAISS-based RAG Pipeline ===")
    print(f"PDF: {pdf_path}")
    print(f"Query: {query}")
    print("-" * 50)
    
    llm = create_llm()
    answer, source_docs, scores = rag_pipeline(llm, pdf_path, query)
    
    print("\n=== Results ===")
    print("Answer:")
    print(answer)
    
    print(f"\nRetrieved {len(source_docs)} source documents:")
    for i, (doc, score) in enumerate(zip(source_docs, scores)):
        print(f"Document {i+1} (similarity: {score:.4f}):")
        print(f"  Page: {doc.metadata.get('page', 'N/A')}")
        print(f"  Content: {doc.page_content[:400]}...")