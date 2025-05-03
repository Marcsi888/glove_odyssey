# backend/app/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes import generation, similarity, sentiment

app = FastAPI(title="GloVe NLP API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(generation.router, prefix="/api/v1/generation")
app.include_router(similarity.router, prefix="/api/v1/similarity")
app.include_router(sentiment.router, prefix="/api/v1/sentiment")