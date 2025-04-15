import requests
import time
import os
import json
try:
    from dotenv import load_dotenv
    has_dotenv = True
except ImportError:
    has_dotenv = False
    print("Note: python-dotenv not installed. Using default configuration.")

try:
    import fix_busted_json
    has_fix_json = True
except ImportError:
    has_fix_json = False

# Load configuration from .env file 
if has_dotenv:
    load_dotenv()

# Ollama local API setup
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
MODEL = os.getenv("OLLAMA_MODEL", "llama3")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))

# Output configuration
SAVE_RESULTS = os.getenv("SAVE_RESULTS", "false").lower() == "true"
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "job_results")

def ask_ollama(prompt, max_retries=3):
    data = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": TEMPERATURE}
    }

    for attempt in range(max_retries):
        try:
            response = requests.post(OLLAMA_URL, json=data)
            response.raise_for_status()
            return response.json()["response"]
        except requests.exceptions.ConnectionError:
            if attempt == max_retries - 1:
                print("\nError: Could not connect to Ollama server.")
                raise Exception("Failed to connect to Ollama server")
            time.sleep(2 ** attempt)  
        except requests.exceptions.RequestException as e:
            if attempt == max_retries - 1:
                raise Exception(f"Failed after {max_retries} retries: {str(e)}")
            time.sleep(2 ** attempt)  
        except (KeyError, json.JSONDecodeError) as e:
            if has_fix_json:
                try:
                    fixed_json = fix_busted_json.fix_busted_json(response.text)
                    return json.loads(fixed_json)["response"]
                except:
                    if attempt == max_retries - 1:
                        raise Exception(f"Failed to parse response: {str(e)}")
                    time.sleep(2 ** attempt)
            else:
                if attempt == max_retries - 1:
                    raise Exception(f"Failed to parse response: {str(e)}")
                time.sleep(2 ** attempt)

# --- Define Agents ---
def Agent_get_job_description(job_title):
    prompt = f"""
    Generate a detailed job description for a {job_title}.
    Include key responsibilities, required skills, and typical industries.
    Format the output clearly with bullet points.
    """
    return ask_ollama(prompt)

def Agent_get_missions_and_tasks(job_description):
    prompt = f"""
    Based on this job description, extract:
    - 3 key missions (numbered)
    - 5 main deliverables (bullet points)
    - 7 critical daily tasks (bullet points)

    Job Description: {job_description}
    """
    return ask_ollama(prompt)

def Agent_get_ai_enhancements(job_description):
    prompt = f"""
    Explain how AI can augment and improve this job role.
    Provide:
    1) Specific AI tools that could be used
    2) Automation opportunities
    3) Efficiency gains
    4) Risks to consider

    Job: {job_description}
    """
    return ask_ollama(prompt)

def Agent_get_technology_recommendations(job_description):
    prompt = f"""
    Based on this job description, recommend specific technologies and tools that would enhance this role.
    For each technology, explain:
    1) What it is and what it does
    2) How it specifically helps with this job role
    3) Approximate learning curve (easy/medium/difficult)
    4) Whether it's free/paid/open-source

    Format as a bulleted list with 5-7 recommendations.

    Job: {job_description}
    """
    return ask_ollama(prompt)

def Agent_get_transition_recommendations(job_title, job_description):
    prompt = f"""
    Provide a detailed transition plan from a traditional {job_title} role
    to an AI-augmented version. Include:

    1) SKILLS TO LEARN:
       - Technical skills needed
       - Soft skills for working with AI
       - Learning resources (courses, books, etc.)

    2) TOOLS TO ADOPT:
       - Essential tools to start with
       - Advanced tools to consider later
       - Integration strategies

    3) MINDSET SHIFTS:
       - From traditional to AI-augmented thinking
       - Overcoming common resistance points
       - Embracing continuous learning

    4) IMPLEMENTATION TIMELINE:
       - First 30 days (quick wins)
       - 2-3 months (building competence)
       - 6-12 months (advanced integration)

    Base your recommendations on this job description: {job_description}
    """
    return ask_ollama(prompt)

def Agent_enhance_job_with_ai(job_title):
    print(f"\nAnalyzing: {job_title}\n")
    results = {}

    try:
        # Agent 1: Job Description
        print("Generating job description :")
        results['job_desc'] = Agent_get_job_description(job_title)

        # Agent 2: Missions & Tasks
        print("Extracting missions and tasks :")
        results['missions_tasks'] = Agent_get_missions_and_tasks(results['job_desc'])

        # Agent 3: Technology Recommendations
        print("Recommending technologies :")
        results['tech_recommendations'] = Agent_get_technology_recommendations(results['job_desc'])

        # Agent 4: AI Enhancements
        print("Identifying AI enhancements :")
        results['ai_enhancements'] = Agent_get_ai_enhancements(results['job_desc'])

        # Agent 5: Transition recommendations
        print("Creating transition plan :")
        results['transition_plan'] = Agent_get_transition_recommendations(job_title, results['job_desc'])

        # Output
        print("\n" + "="*70)
        print(f"**Job Description for {job_title}:**\n{results['job_desc']}\n")
        print(f" **Missions, Deliverables & Tasks:**\n{results['missions_tasks']}\n")
        print(f" **Technology Recommendations:**\n{results['tech_recommendations']}\n")
        print(f" **AI Augmentation Opportunities:**\n{results['ai_enhancements']}\n")
        print(f" **Transition to AI-Augmented Role:**\n{results['transition_plan']}\n")
        print("="*70)

    except Exception as e:
        print(f"\nError : {str(e)}")

def welcome():
    print("\n" + "*"*70)
    print("AI JOB ENHANCEMENT TOOL")
    print("*"*70)
    print("Here, AI anaylzes job and provides AI enhancement recommendations.")
    print(f"Using model: {MODEL} | Temperature: {TEMPERATURE}")

# main program
if __name__ == "__main__":

    welcome()

    while True:
        user_job = input("\nEnter a job title (or 'quit' to exit): ").strip()
        if user_job.lower() in ('end', 'END'):
            print("\nThank you for using the AI Job Enhancement Tool!\n")
            break
        Agent_enhance_job_with_ai(user_job)