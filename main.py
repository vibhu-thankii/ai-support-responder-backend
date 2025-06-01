import json
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# --- Pydantic models for type safety and clear API contracts ---

class QueryRequest(BaseModel):
    text: str

class AIResponse(BaseModel):
    response: str
    confidence: float

# --- Load Knowledge Base on startup ---
try:
    with open("knowledge_base.json", "r") as f:
        knowledge_base = json.load(f)
except (FileNotFoundError, json.JSONDecodeError):
    knowledge_base = []
    print("WARNING: knowledge_base.json not found or invalid. Using fallback only.")


app = FastAPI()

# --- CORS Configuration ---
origins = ["http://localhost:3000","https://ai-support-responder-frontend-fbk6ikkdh-vibhu-thankis-projects.vercel.app"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- The "AI" Logic Function ---
def find_best_response(query_text: str) -> AIResponse:
    """
    Searches the knowledge base and returns a response with a confidence score.
    """
    lower_query = query_text.lower()
    
    best_match = None
    max_matches = 0

    if knowledge_base:
        for entry in knowledge_base:
            matches = sum(1 for keyword in entry["keywords"] if keyword.lower() in lower_query)
            if matches > max_matches:
                max_matches = matches
                best_match = entry

    if best_match and max_matches > 0:
        # Calculate a simple confidence score
        # Confidence = (number of matched keywords / total keywords for that rule)
        # We add a small base confidence to make it look better
        total_keywords = len(best_match["keywords"])
        confidence = 0.5 + (max_matches / total_keywords) * 0.5 
        return AIResponse(response=best_match["response"], confidence=min(confidence, 0.98))
    
    # A generic fallback response with low confidence
    fallback_response = "Thank you for your query. A member of our team will review your message and get back to you shortly. We appreciate your patience."
    return AIResponse(response=fallback_response, confidence=0.30)


@app.get("/")
def read_root():
    return {"status": "AI Responder Backend is running."}

@app.post("/api/generate-response", response_model=AIResponse)
async def generate_response(request: QueryRequest):
    """
    Receives a customer query and returns a generated draft and confidence score.
    """
    return find_best_response(request.text)
