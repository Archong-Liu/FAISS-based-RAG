from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import backend.pipeline as pipeline
import os

# Create LLM
print("Initializing LLM...")
llm = pipeline.create_llm()


app = FastAPI(title="FAISS-based RAG API", description="RAG pipeline with FAISS and llama.cpp")

# 添加 CORS 支援，允許前端訪問
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 在生產環境中應該設定具體的域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 定義請求模型
class QueryRequest(BaseModel):
    query: str
    pdf_path: str = "data/Sample.pdf"  # 可選，預設使用 Sample.pdf

# 定義回應模型
class QueryResponse(BaseModel):
    answer: str
    source_documents: list
    similarity_scores: list

@app.get("/")
async def root():
    return {"message": "FAISS-based RAG API is running!"}

@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    try:
        # 檢查 PDF 檔案是否存在
        if not os.path.exists(request.pdf_path):
            raise HTTPException(status_code=404, detail=f"PDF file not found: {request.pdf_path}")
        
        # 執行 RAG pipeline
        answer, source_docs, scores = pipeline.rag_pipeline(llm, request.pdf_path, request.query)
        
        # 格式化回應
        source_documents = []
        for doc in source_docs:
            source_documents.append({
                "content": doc.page_content,
                "metadata": doc.metadata
            })
        
        return QueryResponse(
            answer=answer,
            source_documents=source_documents,
            similarity_scores=scores
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")
                

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)