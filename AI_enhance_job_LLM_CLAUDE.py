import time
import os
import json
import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

try:
    from dotenv import load_dotenv
    has_dotenv = True
except ImportError:
    has_dotenv = False
    print("Note: python-dotenv not installed. Using default configuration.")

# Load configuration from .env file
if has_dotenv:
    load_dotenv()

# AWS Bedrock Configuration
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID", "")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", "")
CLAUDE_MODEL_ID = os.getenv("CLAUDE_MODEL_ID", "anthropic.claude-3-sonnet-20240229-v1:0")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "4000"))

# Configure AWS Bedrock client
bedrock_config = Config(
    region_name=AWS_REGION,
    retries={
        'max_attempts': 5,
        'mode': 'standard'
    }
)

bedrock = boto3.client(
    service_name='bedrock-runtime',
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    config=bedrock_config
)

def claude(prompt, max_retries=3):
    """
    Invokes Claude model through AWS Bedrock
    """
    if not AWS_ACCESS_KEY_ID or not AWS_SECRET_ACCESS_KEY:
        raise Exception("AWS credentials not found. Please add your AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY to the .env file.")

    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        ]
    })

    for attempt in range(max_retries):
        try:
            response = bedrock.invoke_model(
                body=body,
                modelId=CLAUDE_MODEL_ID
            )
            response_body = json.loads(response.get('body').read())
            return response_body['content'][0]['text']
        except ClientError as e:
            if attempt == max_retries - 1:
                raise Exception(f"Failed after {max_retries} retries: {str(e)}")
            time.sleep(2 ** attempt)
        except Exception as e:
            if attempt == max_retries - 1:
                raise Exception(f"Failed to parse response: {str(e)}")
            time.sleep(2 ** attempt)

# --- Define Agents ---
def Agent_get_job_description(job_title):
    prompt = f"""
    Generate a concise job description for a {job_title}.
    Include key responsibilities, required skills, and typical industries.
    Format the output clearly with bullet points.
    Keep your response under 200 words and focus on the most essential information.
    Use short, clear sentences and avoid unnecessary jargon.
    """
    return claude(prompt)

def Agent_get_missions_and_tasks(job_description):
    prompt = f"""
    Using ONLY the job description provided below, extract:
    - 3 key missions (numbered) - one sentence each
    - 5 main deliverables (bullet points) - one sentence each
    - 7 critical daily tasks (bullet points) - keep to 5-7 words each

    Do not add any information that is not directly derived from the job description.
    Keep your entire response under 250 words.
    Use clear, direct language and avoid unnecessary elaboration.

    Job Description: {job_description}
    """
    return claude(prompt)

def Agent_get_ai_enhancements(job_description):
    prompt = f"""
    Concisely explain how AI can augment and improve this job role.
    Provide in bullet point format:
    1) Specific AI tools that could be used (3-4 tools, one sentence each)
    2) Automation opportunities (3-4 points, one sentence each)
    3) Efficiency gains (3-4 points, one sentence each)
    4) Risks to consider (3-4 points, one sentence each)

    Keep your entire response under 300 words.
    Use clear, direct language with no unnecessary elaboration.
    Focus on practical, actionable insights rather than theoretical possibilities.

    Job: {job_description}
    """
    return claude(prompt)

def Agent_get_technology_recommendations(job_description):
    prompt = f"""
    Based on this job description, recommend 5 specific technologies and tools that would enhance this role.
    For each technology, provide a single concise paragraph (2-3 sentences) that includes:
    1) What it is and what it does
    2) How it specifically helps with this job role
    3) Approximate learning curve (easy/medium/difficult)
    4) Whether it's free/paid/open-source

    Format as a bulleted list with exactly 5 recommendations.
    Keep your entire response under 300 words.
    Focus on the most impactful technologies rather than covering everything possible.

    Job: {job_description}
    """
    return claude(prompt)

def Agent_get_transition_recommendations(job_title, job_description, ai_enhancements=None, tech_recommendations=None):
    # Prepare additional context if available
    additional_context = ""
    if ai_enhancements:
        additional_context += f"\n\nAI ENHANCEMENT OPPORTUNITIES IDENTIFIED:\n{ai_enhancements}"
    if tech_recommendations:
        additional_context += f"\n\nRECOMMENDED TECHNOLOGIES:\n{tech_recommendations}"

    prompt = f"""
    You are a specialized AI transformation consultant with expertise in helping professionals transition to AI-augmented roles.
    Your task is to create a concise, practical, and actionable transition plan for a {job_title} to evolve into an AI-augmented professional.

    First, analyze this job description carefully: {job_description}
    {additional_context}

    Then, create a focused transition roadmap with the following sections, keeping the ENTIRE response under 600 words:

    1) SKILLS DEVELOPMENT PLAN (25% of your response):
       - TECHNICAL SKILLS: List 3 specific technical skills most relevant for this role. For each, provide a one-sentence explanation of importance.
       - SOFT SKILLS: List 3 critical soft skills needed when working with AI. One sentence each.
       - LEARNING RESOURCES: For each skill, recommend ONE specific resource (course, book, or certification).

    2) AI TOOLS IMPLEMENTATION STRATEGY (25% of your response):
       - IMMEDIATE ADOPTION (First 30 days): List 2 user-friendly AI tools with one-sentence descriptions.
       - INTERMEDIATE ADOPTION (2-3 months): List 2 more advanced tools with one-sentence descriptions.
       - ADVANCED ADOPTION (6-12 months): List 1 sophisticated AI solution with a one-sentence description.
       - For each tool, only note whether it's free/paid/open-source.

    3) PSYCHOLOGICAL & ORGANIZATIONAL ADAPTATION (25% of your response):
       - MINDSET EVOLUTION: 2-3 sentences on required mindset shifts.
       - RESISTANCE MANAGEMENT: List 2 common resistance points with one-sentence strategies to overcome each.
       - ETHICAL CONSIDERATIONS: List 1 key ethical consideration with a one-sentence recommendation.

    4) PHASED IMPLEMENTATION PLAN (25% of your response):
       - FIRST 30 DAYS: 2-3 bullet points with specific goals.
       - 2-3 MONTHS: 2-3 bullet points with specific goals.
       - 6-12 MONTHS: 2-3 bullet points with specific goals.
       - SUCCESS METRICS: List 3 specific metrics (one sentence each).

    Format your response with clear headings and bullet points. Use extremely concise language. Make all recommendations highly specific to the {job_title} role, not generic advice. Prioritize brevity and clarity over comprehensiveness.
    """
    return claude(prompt)

def Agent_enhance_job_with_ai(job_title):
    print(f"\nAnalyzing: {job_title}\n")
    results = {}

    try:
        # Agent 1: Job Description
        print("Generating job description...")
        results['job_desc'] = Agent_get_job_description(job_title)

        # Agent 2: Missions & Tasks
        print("Extracting missions and tasks...")
        results['missions_tasks'] = Agent_get_missions_and_tasks(results['job_desc'])

        # Agent 3: Technology Recommendations
        print("Recommending technologies...")
        results['tech_recommendations'] = Agent_get_technology_recommendations(results['job_desc'])

        # Agent 4: AI Enhancements
        print("Identifying AI enhancements...")
        results['ai_enhancements'] = Agent_get_ai_enhancements(results['job_desc'])

        # Agent 5: Transition recommendations
        print("Creating transition plan...")
        results['transition_plan'] = Agent_get_transition_recommendations(
            job_title,
            results['job_desc'],
            ai_enhancements=results['ai_enhancements'],
            tech_recommendations=results['tech_recommendations']
        )

        # Output
        print("\n" + "*"*70)
        print(f"**Job Description for {job_title}:**\n{results['job_desc']}\n")
        print(f"**Missions, Deliverables & Tasks:**\n{results['missions_tasks']}\n")
        print(f"**Technology Recommendations:**\n{results['tech_recommendations']}\n")
        print(f"**AI Augmentation Opportunities:**\n{results['ai_enhancements']}\n")
        print(f"**Transition to AI-Augmented Role:**\n{results['transition_plan']}\n")
        print("*"*70)

    except Exception as e:
        print(f"\nError: {str(e)}")

def welcome():
    print("\n" + "*"*70)
    print("AI JOB ENHANCEMENT TOOL (AWS BEDROCK CLAUDE VERSION)")
    print("*"*70)
    print("Analyzes jobs and provides AI enhancement recommendations using AWS Bedrock.")
    print(f"Using model: {CLAUDE_MODEL_ID} | Temperature: {TEMPERATURE}")

# main program
if __name__ == "__main__":
    welcome()

    while True:
        user_job = input("\nEnter a job title (or 'end' to exit): ").strip()
        if user_job.lower() in ('END', 'end'):
            print("\nThank you for using the AI Job Enhancement Tool!\n")
            break
        Agent_enhance_job_with_ai(user_job)