from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes import generation, similarity, sentiment, analysis
import uvicorn

app = FastAPI(
    title="GloVe NLP - The Odyssey Analysis",
    description="Advanced NLP analysis of classical texts using GloVe embeddings",
    version="1.0.0"
)

# CORS added for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# presenting route modules
app.include_router(generation.router, prefix="/api/v1/generation", tags=["text-generation"])
app.include_router(similarity.router, prefix="/api/v1/similarity", tags=["word-similarity"])
app.include_router(sentiment.router, prefix="/api/v1/sentiment", tags=["sentiment-analysis"])
app.include_router(analysis.router, prefix="/api/v1/analysis", tags=["text-analysis"])

@app.get("/")
async def root():
    return {"message": "GloVe NLP API for The Odyssey - Welcome to the classical text analysis system!"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "api_version": "1.0.0"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)