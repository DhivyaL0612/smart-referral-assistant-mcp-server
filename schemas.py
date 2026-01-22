#create data structure of what goes in and out using basemodel

from pydantic import BaseModel, Field
from typing import List


# Schema for the data the user will send to api
class ProfileAnalysisRequest(BaseModel):
    profile_text: str = Field(
        ..., # The '...' means this field is required
        min_length=50,
        description="The full text of the user's profile or bio."
    )
# Schema for the data our API will return
class ProfileAnalysisResponse(BaseModel):
    skills: List[str] = Field(..., description="A list of key skills extracted from the profile.")
    years_of_experience: int = Field(..., description="The estimated total years of professional experience.")


# Schema for the data the user will send to api
class JobDeconstructionRequest(BaseModel):
    job_description_text: str = Field(
        ..., # The '...' means this field is required
        min_length=50,
        description="The full text of the job description."
    )
# Schema for the data our API will return
class JobDeconstructionResponse(BaseModel):
    key_requirements: List[str] = Field(..., description="A list of the top 5-7 most critical requirements from the job description.")
    job_title: str = Field(..., description="The job title extracted from the description.")


# Schema for the final referral generation endpoint
class ReferralGenerationRequest(BaseModel):
    # We are nesting our previous response schemas here!
    analyzed_profile: ProfileAnalysisResponse
    deconstructed_job: JobDeconstructionResponse

class ReferralGenerationResponse(BaseModel):
    referral_message: str = Field(..., description="The final, generated referral request message.")


