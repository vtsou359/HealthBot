import os
from typing import TypedDict, List, Dict, Any, Optional, Literal
from dotenv import load_dotenv

# LangChain and LangGraph imports
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults

# Gradio for UI
import gradio as gr

# Load environment variables
load_dotenv()

# Initialize the language model
llm = ChatOpenAI(
    model="gpt-4.1",
    temperature=0.0,
)
# Initialize the Tavily search tool
search_tool = TavilySearchResults(max_results=5)



######### Define the state schema #########

class HealthBotState(TypedDict):
    # User inputs
    health_topic: Optional[str]
    quiz_ready: Optional[bool]
    quiz_answer: Optional[str]
    next_action: Optional[Literal["new_topic", "exit", "more_questions"]]
    difficulty: Optional[Literal["easy", "medium", "hard"]]
    level_of_details: Optional[Literal["easy", "medium", "hard"]]  # For summarization
    num_questions: Optional[int]
    current_question_index: Optional[int]

    # System outputs
    search_results: Optional[List[Dict[str, Any]]]
    summary: Optional[str]
    quiz_questions: Optional[List[str]]
    current_quiz_question: Optional[str]
    quiz_grade: Optional[str]
    quiz_feedback: Optional[str]
    quiz_grades: Optional[List[str]]  # Store grades for all questions
    related_topics: Optional[List[str]]

    # Messages for the conversation
    messages: List[Dict[str, Any]]

######### ######### ######### #########


######### Define the nodes for the workflow #########

def ask_health_topic(state: HealthBotState) -> HealthBotState:
    """
    Asks the user for a health topic and difficulty level, updating the `HealthBotState`
    accordingly. This function initializes or resets conversation state as needed, and
    prompts the user for input. It ensures the conversation includes initial system
    and assistant messages, user responses, and updated state metadata such as the
    health topic and difficulty level.

    :param state: The current state of the HealthBot, containing conversation messages
                  and metadata such as the user's chosen health topic and difficulty level.
                  If `state` is newly initialized or signals a new topic request via
                  `next_action`, the conversation is reset.

    :return: The updated `HealthBotState` object with new user-provided health topic,
             difficulty level, and conversation messages appended.
    """
    # If this is a new conversation or the user wants to learn about a new topic
    if state.get("health_topic") is None or state.get("next_action") == "new_topic":
        # Reset the state for a new topic
        if state.get("next_action") == "new_topic":
            state = HealthBotState(messages=state.get("messages", []))

        # Add system message to the conversation
        if not state.get("messages"):
            state["messages"] = [
                {
                    "role": "system",
                    "content": "You are HealthBot, an AI assistant that helps patients learn about health topics."
                }
            ]

        # Add the question to the conversation
        state["messages"].append({
            "role": "assistant",
            "content": "What health topic or medical condition would you like to learn about today?"
        })

        # This question now appears automatically as the first message when opening the Gradio UI

        # Get user input
        health_topic = input("HealthBot: What health topic or medical condition would you like to learn about today? ")

        # Add user response to the conversation
        state["messages"].append({
            "role": "user",
            "content": health_topic
        })

        # Update the state
        state["health_topic"] = health_topic

        # Ask for difficulty level
        state["messages"].append({
            "role": "assistant",
            "content": "What level of detail would you like? (easy, medium, hard)"
        })

        difficulty = input("HealthBot: What level of detail would you like? (easy, medium, hard) ")

        # Add user response to the conversation
        state["messages"].append({
            "role": "user",
            "content": difficulty
        })

        # Update the state
        state["difficulty"] = difficulty
        state["level_of_details"] = difficulty  # Set level_of_details from difficulty

    return state



#2.