from fastapi import FastAPI
from pydantic import BaseModel

from src.rag.retriever import retrieve_context
from src.llm.gemini_client import generate_answer

app = FastAPI(title="DocuMind AI API")

class Question(BaseModel):
    query: str

@app.get("/")
def root():
    return {"message": "DocuMind AI API is running"}

@app.post("/ask")
def ask(question: Question):
    context = retrieve_context(question.query)
    prompt = f"Context: {context}\nQuestion: {question.query}"
    answer = generate_answer(prompt)

    return {
        "question": question.query,
        "answer": answer
    }
