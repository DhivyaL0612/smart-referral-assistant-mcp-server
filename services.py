import os
import schemas as schemas 


# LangChain specific imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.output_parsers import StrOutputParser


# We tell the parser what our desired output schema is.
parser = JsonOutputParser(pydantic_object=schemas.ProfileAnalysisResponse)

api_key = os.getenv("GEMINI_KEY")
# llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", api_key = api_key)

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.2,
    google_api_key=api_key # Explicitly passing the key is good practice
)


# 3. Create a LangChain Prompt Template
# Notice the "{format_instructions}" placeholder. This is where the parser
# will automatically inject instructions on how to format the JSON.
prompt = PromptTemplate(
    template="""
        Analyze the following user profile text. Your task is to extract key information based on the user's request.
        Do not include any introductory text, markdown formatting (like ```json), or explanations.
        Your output must be a single, valid JSON object that strictly follows the provided schema.

        {format_instructions}

        ---
        Profile Text:
        {profile_text}
        ---
    """,
    input_variables=["profile_text"],
    # This connects the prompt to our desired JSON output format
    partial_variables={"format_instructions": parser.get_format_instructions()},
)


#langchain chain

chain = prompt | llm | parser

# 5. Define our service function
def analyze_user_profile(request: schemas.ProfileAnalysisRequest) -> schemas.ProfileAnalysisResponse:
    """
    Takes user profile text and uses a LangChain chain to extract structured data.
    """
    # Invoke the chain with the user's profile text
    # The 'chain.invoke' method handles everything: formatting the prompt,
    # calling the model, and parsing the output into a dictionary.
    extracted_data = chain.invoke({"profile_text": request.profile_text})

    # The 'extracted_data' is already a dictionary, so we can directly
    # validate it with our Pydantic schema before returning.
    return schemas.ProfileAnalysisResponse(**extracted_data)

jobdesc_parser = JsonOutputParser(pydantic_object=schemas.JobDeconstructionResponse)


job_prompt = PromptTemplate(
    template="""
        Analyze the following job description text. Your primary goal is to extract the most critical requirements and the job title.
        Your output must be a single, valid JSON object that strictly follows the provided schema. Do not add any extra commentary.

        {format_instructions}

        ---
        Job Description Text:
        {job_description_text}
        ---
    """,
    input_variables=["job_description_text"],
    partial_variables={"format_instructions": jobdesc_parser.get_format_instructions()},
)

# 3. Create the new chain (it can reuse the same llm object)
job_chain = job_prompt | llm | jobdesc_parser

def deconstruct_job_description(request: schemas.JobDeconstructionRequest) -> schemas.JobDeconstructionResponse:
    """
    Takes job description text and uses a LangChain chain to extract structured data.
    """
    extracted_data = job_chain.invoke({"job_description_text": request.job_description_text})
    return schemas.JobDeconstructionResponse(**extracted_data)



referral_prompt = PromptTemplate.from_template(
    """
    You are an expert career assistant. Your task is to write a concise, professional, and compelling referral request message.
    The user will provide their analyzed profile and the key requirements of a job they are interested in.
    Your goal is to bridge the two, highlighting the strongest points of alignment.

    **Instructions:**
    1.  Start with a polite and direct opening.
    2.  In the main body, explicitly mention 2-3 key skills from the user's profile that directly match the job's key requirements.
    3.  Subtly mention the user's years of experience to establish seniority.
    4.  Maintain a professional and enthusiastic tone.
    5.  Keep the entire message concise, ideally under 150 words.
    6.  Do not include any placeholders like "[Your Name]" or "[Hiring Manager Name]". The user will add those themselves.
    7.  Your output must be only the text of the message, with no extra commentary or titles.

    **Analyzed User Profile:**
    - Skills: {skills}
    - Years of Experience: {years_of_experience}

    **Deconstructed Job Description:**
    - Job Title: {job_title}
    - Key Requirements: {key_requirements}

    ---
    Generate the referral message now:
    """
)

referral_chain = referral_prompt | llm | StrOutputParser()

def generate_referral_message(request: schemas.ReferralGenerationRequest) -> schemas.ReferralGenerationResponse:
    """
    Takes the analyzed profile and job, and generates a compelling referral request message.
    """
    # We need to flatten the nested data to pass it to the chain.
    chain_input = {
        "skills": request.analyzed_profile.skills,
        "years_of_experience": request.analyzed_profile.years_of_experience,
        "job_title": request.deconstructed_job.job_title,
        "key_requirements": request.deconstructed_job.key_requirements
    }

    message = referral_chain.invoke(chain_input)

    return schemas.ReferralGenerationResponse(referral_message=message)

