from fastapi import FastAPI
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel
import mcp_server.schemas as schemas
import mcp_server.services as services



#fast api instance

app = FastAPI(title="Smart Referral assistant - MCP Server",
              description="The Model-Context-Protocol server for handling all AI interactions.", 
              version="0.1.0")


@app.get("/")

def read_root():
    """A simple endpoint to check if the server is running."""
    return {"status": "MCP Server is running!"}


@app.post("/analyze_profile", response_model=schemas.ProfileAnalysisResponse)
def analyze_profile_endpoint(request: schemas.ProfileAnalysisRequest):
    """
    Receives user profile text and returns a structured analysis
    of skills and experience.
    """

    return services.analyze_user_profile(request)

@app.post("/deconstruct_job", response_model=schemas.JobDeconstructionResponse)
def deconstruct_job_endpoint(request: schemas.JobDeconstructionRequest):
    """
    Receives user profile text and returns a structured analysis
    of skills and experience.
    """

    return services.deconstruct_job_description(request)

@app.post("/generate_referral_request", response_model=schemas.ReferralGenerationResponse)
def generate_referral_endpoint(request: schemas.ReferralGenerationRequest):
    """
    Takes structured profile and job data and returns a generated
    referral request message.
    """
    return services.generate_referral_message(request)