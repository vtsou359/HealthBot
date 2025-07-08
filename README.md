# HealthBot: AI-Powered Patient Education System

HealthBot is an AI-powered chatbot designed to provide personalized, on-demand health information to patients. It helps patients understand their medical conditions, treatment options, and post-treatment care instructions through interactive conversations, summaries, and quizzes.

## Features

- **Health Topic Search**: Ask about any health topic or medical condition
- **Personalized Summaries**: Get patient-friendly summaries of health information
- **Comprehension Quizzes**: Test your understanding with customizable quizzes
- **Difficulty Settings**: Choose between easy, medium, or hard levels of detail
- **Multiple Quiz Questions**: Select how many questions you want in your quiz
- **Related Topics**: Discover related health topics based on your interests

## Requirements

- Python 3.12+
- OpenAI API key
- Tavily API key

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/HealthBot.git
   cd HealthBot
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

3. Create a `.env` file in the project root with your API keys:
   ```
   OPENAI_API_KEY="your_openai_api_key"
   TAVILY_API_KEY="your_tavily_api_key"
   ```

## Usage

### Gradio Web Interface

Run the Gradio web interface:

```
python src/healthbot.py
```

This will start a local web server with the HealthBot interface. You can:
- Type any health topic to learn about it
- Select the difficulty level (easy, medium, hard)
- Choose how many quiz questions you want
- Take quizzes to test your understanding
- Explore related health topics

### Terminal Interface

You can also run HealthBot in the terminal. Open the `src/healthbot.py` file and uncomment the last line to use the terminal interface:

```python
if __name__ == "__main__":
    # Launch the Gradio interface
    # demo.launch()
    
    # Run the terminal version
    run_healthbot()
```

## How It Works

HealthBot uses LangGraph to create a workflow with the following steps:

1. Ask the patient what health topic they'd like to learn about
2. Search for information using Tavily, focusing on reputable medical sources
3. Summarize the results in patient-friendly language
4. Present the summary to the patient
5. Create and present quiz questions based on the summary
6. Grade the patient's answers and provide feedback
7. Suggest related health topics
8. Allow the patient to learn about a new topic or exit

## Project Structure

- `src/healthbot.py`: Main implementation of the HealthBot using LangGraph
- `requirements.txt`: Required Python packages
- `.env`: Environment variables for API keys
- `experimental_ntbks/`: Example notebooks for reference
