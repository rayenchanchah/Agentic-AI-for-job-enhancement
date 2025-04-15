# AI Job Enhancement Tool

## Overview

The AI Job Enhancement Tool is a Python application that uses Large Language Models (LLMs) to analyze job roles and provide AI-enhanced recommendations for transforming traditional job roles into AI-augmented versions. The tool generates comprehensive reports covering job descriptions, key missions, technology recommendations, AI augmentation opportunities, and transition plans.

## Features

- **Job Analysis**: Generates detailed job descriptions for any role
- **Mission Extraction**: Identifies key missions, deliverables, and tasks
- **Technology Recommendations**: Suggests specific tools and technologies
- **AI Enhancement**: Provides AI augmentation opportunities
- **Transition Planning**: Creates detailed transition plans to AI-augmented roles
- **Result Export**: Saves analysis results to markdown files (optional)

## Requirements

- Python 3.6+
- Ollama (for local LLM inference)

### Optional Dependencies

- `python-dotenv`: For configuration via .env file
- `requests`: For making API calls to Ollama

## Installation

1. Clone this repository or download the source files
2. Install Ollama from [ollama.com](https://ollama.com/)
3. Pull the Llama 3 model (or another model of your choice):
   ```
   ollama pull llama3
   ```
4. Install required dependencies:
   ```
   pip install requests python-dotenv
   ```

## Configuration

Create a `.env` file in the project directory with the following options:

```
# Ollama Configuration
OLLAMA_URL=http://localhost:11434/api/generate
OLLAMA_MODEL=llama3
TEMPERATURE=0.7

# Output Configuration
SAVE_RESULTS=false
OUTPUT_DIR=job_results
```

## Usage

### Running the Python Script

Run the script:

```
python AI_enhance_job.py
```

Enter a job title when prompted, and the tool will generate a comprehensive analysis.

## Example Output

The tool provides:

1. **Job Description**: Detailed responsibilities and requirements
2. **Missions & Tasks**: Key missions, deliverables, and daily tasks
3. **Technology Recommendations**: Specific tools to enhance the role
4. **AI Augmentation**: Opportunities for AI integration
5. **Transition Plan**: Step-by-step guide to transform the role

## Technical Details

### Architecture

The tool uses a modular architecture with several key components:

1. **Configuration Module**: Handles environment variables and settings
2. **LLM Interface**: Communicates with the Ollama API
3. **Agent Functions**: Specialized prompts for different aspects of job analysis
4. **Output Formatting**: Displays and saves results

### Prompt Engineering

The tool uses carefully crafted prompts to extract specific information:

- **Job Description Prompt**: Extracts responsibilities, skills, and industries
- **Missions & Tasks Prompt**: Identifies key missions, deliverables, and tasks
- **Technology Recommendations Prompt**: Suggests relevant tools and technologies
- **AI Enhancements Prompt**: Identifies AI augmentation opportunities
- **Transition Plan Prompt**: Creates a detailed transition roadmap

## Project Files

- `AI_enhance_job.py`: Main Python script
- `.env`: Configuration file (optional)
- `README.md`: Project documentation

## Future Enhancements

Potential future improvements include:

- Support for additional LLM providers (OpenAI, Hugging Face, etc.)
- Interactive web interface
- Industry-specific job analysis
- Comparative analysis between different roles
- Integration with job posting platforms

## License

MIT

## Credits

Developed as part of an internship project at EDLIGO.
