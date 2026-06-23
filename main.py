from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai
import json
import os

app = FastAPI()

# Enable CORS so your teammates' frontend/backend can communicate with your API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure Gemini API using your secret key
# (We will set this securely on Render later so no one can steal your key)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# Define what data the backend must send us (Scenario and Answer)
class EvaluationRequest(BaseModel):
    scenario: str
    answer: str

@app.get("/")
def home():
    return {"message": "Skillverse AI Assessment API is running smoothly!"}

@app.post("/evaluate")
async def evaluate_skills(request: EvaluationRequest):
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="Gemini API Key is missing on the server.")
    
    try:
        # The instruction prompt that tells Gemini exactly how to grade the answer
        system_instruction = (
            "You are an expert innovation skills assessor. Evaluate the student's open-ended answer "
            "based on the provided scenario challenge. You must evaluate five specific competencies: "
            "Critical Thinking, Problem Solving, Communication, Creativity, and Collaboration.\n\n"
            "You MUST respond ONLY with a valid JSON object matching this exact camelCase structure, "
            "with no extra conversational markdown text, formatting, or backticks:\n"
            "{\n"
            "  \"criticalThinking\": { \"score\": 1-10, \"performance\": \"reason for score\" },\n"
            "  \"problemSolving\": { \"score\": 1-10, \"performance\": \"reason for score\" },\n"
            "  \"communication\": { \"score\": 1-10, \"performance\": \"reason for score\" },\n"
            "  \"creativity\": { \"score\": 1-10, \"performance\": \"reason for score\" },\n"
            "  \"collaboration\": { \"score\": 1-10, \"performance\": \"reason for score\" },\n"
            "  \"overallScore\": 1-10,\n"
            "  \"actionableFeedback\": {\n"
            "    \"strengths\": [\"strength 1\", \"strength 2\"],\n"
            "    \"weaknesses\": [\"weakness 1\"],\n"
            "    \"improvementSuggestions\": \"metacognitive coaching suggestion\"\n"
            "  }\n"
            "}"
        )
        
        # Call the Gemini model
        model = genai.GenerativeModel("gemini-1.5-flash")
        user_prompt = f"Scenario Challenge: {request.scenario}\nStudent's Answer: {request.answer}"
        
        response = model.generate_content(
            f"{system_instruction}\n\nEvaluate this input:\n{user_prompt}"
        )
        
        # Clean the response text to ensure it's pure JSON
        cleaned_text = response.text.strip().replace("```json", "").replace("```", "").strip()
        
        # Convert text to actual JSON data
        evaluation_data = json.loads(cleaned_text)
        return evaluation_data

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during AI evaluation: {str(e)}")
