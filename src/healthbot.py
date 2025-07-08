"""
HealthBot: An AI-Powered Patient Education System

This module implements a LangGraph workflow for a healthcare chatbot that can:
1. Ask patients about health topics they want to learn about
2. Search for information using Tavily
3. Summarize the information in patient-friendly language
4. Create and grade comprehension quizzes
5. Provide feedback and suggestions for related topics

The implementation includes a Gradio UI for easy interaction.
"""

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

# Define the state schema
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

# Define the nodes for the workflow


def ask_health_topic(state: HealthBotState) -> HealthBotState:
    """Ask the patient what health topic they'd like to learn about."""
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

def search_health_info(state: HealthBotState) -> HealthBotState:
    """Search for health information using Tavily."""
    health_topic = state["health_topic"]
    level_of_details = state.get("level_of_details", state.get("difficulty", "medium"))

    # Construct the search query based on level_of_details
    if level_of_details == "easy":
        query = f"{health_topic} simple explanation for patients"
    elif level_of_details == "hard":
        query = f"{health_topic} detailed medical information"
    else:  # medium
        query = f"{health_topic} patient information"

    # Search for information
    search_results = search_tool.invoke(query)

    # Update the state
    state["search_results"] = search_results

    return state

def summarize_health_info(state: HealthBotState) -> HealthBotState:
    """Summarize the health information in patient-friendly language."""
    search_results = state["search_results"]
    health_topic = state["health_topic"]
    level_of_details = state.get("level_of_details", state.get("difficulty", "medium"))

    # Create a prompt for the LLM to summarize the information
    prompt = f"""
    You are a healthcare educator explaining {health_topic} to a patient.

    Based on the following search results, create a {level_of_details} level summary about {health_topic}.

    If the level is 'easy', use simple language, avoid medical jargon, and keep it brief (2-3 paragraphs).
    If the level is 'medium', use moderately complex language and provide more details (3-4 paragraphs).
    If the level is 'hard', use more technical language and provide comprehensive information (4-5 paragraphs).

    Search Results:
    {search_results}

    Your summary should be informative, accurate, and helpful for a patient trying to understand this health topic.
    Include important facts, symptoms, treatments, and preventive measures when applicable.
    """

    # Generate the summary
    messages = [
        SystemMessage(content="You are a healthcare educator explaining medical topics to patients."),
        HumanMessage(content=prompt)
    ]
    response = llm.invoke(messages)

    # Update the state
    state["summary"] = response.content

    # Add the summary to the conversation
    state["messages"].append({
        "role": "assistant",
        "content": response.content
    })

    return state

def present_summary(state: HealthBotState) -> HealthBotState:
    """Present the summary to the patient."""
    summary = state["summary"]

    # Print the summary
    print(f"HealthBot: Here's what I found about {state['health_topic']}:\n")
    print(summary)
    print("\n")

    return state

def prompt_for_quiz(state: HealthBotState) -> HealthBotState:
    """Ask if the patient is ready for a comprehension check."""
    # Add the question to the conversation
    state["messages"].append({
        "role": "assistant",
        "content": "Would you like to take a quick quiz to test your understanding? (yes/no)"
    })

    # Get user input
    response = input("HealthBot: Would you like to take a quick quiz to test your understanding? (yes/no) ")

    # Add user response to the conversation
    state["messages"].append({
        "role": "user",
        "content": response
    })

    # Update the state
    state["quiz_ready"] = response.lower() in ["yes", "y", "sure", "ok", "okay"]

    if state["quiz_ready"]:
        # Ask for number of questions
        state["messages"].append({
            "role": "assistant",
            "content": "How many questions would you like? (1-5)"
        })

        num_questions = input("HealthBot: How many questions would you like? (1-5) ")

        # Add user response to the conversation
        state["messages"].append({
            "role": "user",
            "content": num_questions
        })

        # Update the state
        try:
            state["num_questions"] = min(5, max(1, int(num_questions)))
        except ValueError:
            state["num_questions"] = 1
            print("HealthBot: I'll ask you 1 question.")

        state["current_question_index"] = 0

    return state

def create_quiz_questions(state: HealthBotState) -> HealthBotState:
    """Create quiz questions based on the health information summary."""
    if not state["quiz_ready"]:
        return state

    summary = state["summary"]
    health_topic = state["health_topic"]
    difficulty = state.get("difficulty", "medium")
    num_questions = state.get("num_questions", 1)

    # Create a prompt for the LLM to generate quiz questions
    prompt = f"""
    Based on the following summary about {health_topic}, create {num_questions} quiz question(s) to test the patient's understanding.

    Summary:
    {summary}

    If the difficulty is 'easy', create straightforward questions with clear answers from the summary.
    If the difficulty is 'medium', create questions that require some synthesis of information.
    If the difficulty is 'hard', create questions that require deeper understanding and application of concepts.

    Format your response as a JSON array of strings, with each string being a question.
    """

    # Generate the quiz questions
    messages = [
        SystemMessage(content="You are creating quiz questions to test patient understanding of medical topics."),
        HumanMessage(content=prompt)
    ]
    response = llm.invoke(messages)

    # Parse the response to extract the questions
    # This is a simple approach; in a production system, you'd want more robust parsing
    import json
    try:
        questions_text = response.content
        # Find the JSON array in the response
        start_idx = questions_text.find('[')
        end_idx = questions_text.rfind(']') + 1
        if start_idx >= 0 and end_idx > start_idx:
            questions_json = questions_text[start_idx:end_idx]
            questions = json.loads(questions_json)
        else:
            # Fallback if JSON parsing fails
            questions = [questions_text]
    except:
        # Fallback if JSON parsing fails
        questions = [response.content]

    # Update the state
    state["quiz_questions"] = questions
    state["current_quiz_question"] = questions[0]

    return state

def present_quiz_question(state: HealthBotState) -> HealthBotState:
    """Present a quiz question to the patient."""
    if not state["quiz_ready"]:
        return state

    current_index = state.get("current_question_index", 0)
    questions = state["quiz_questions"]

    if current_index < len(questions):
        current_question = questions[current_index]

        # Add the question to the conversation
        state["messages"].append({
            "role": "assistant",
            "content": f"Question {current_index + 1}: {current_question}"
        })

        # Print the question
        print(f"HealthBot: Question {current_index + 1}: {current_question}")

        # Update the state
        state["current_quiz_question"] = current_question

    return state

def collect_quiz_answer(state: HealthBotState) -> HealthBotState:
    """Collect the patient's answer to the quiz question."""
    if not state["quiz_ready"]:
        return state

    # Get user input
    answer = input("Your answer: ")

    # Add user response to the conversation
    state["messages"].append({
        "role": "user",
        "content": answer
    })

    # Update the state
    state["quiz_answer"] = answer

    return state

def grade_quiz_answer(state: HealthBotState) -> HealthBotState:
    """Grade the patient's answer to the quiz question."""
    if not state["quiz_ready"]:
        return state

    question = state["current_quiz_question"]
    answer = state["quiz_answer"]
    summary = state["summary"]

    # Create a prompt for the LLM to grade the answer
    prompt = f"""
    Grade the patient's answer to the following question about the health topic.

    Question: {question}

    Patient's Answer: {answer}

    Information from the summary:
    {summary}

    Provide a grade 'Pass' or 'Fail' and detailed feedback explaining why the answer received that grade.
    Include specific information from the summary that supports or contradicts the patient's answer.
    Be encouraging and educational in your feedback.

    Format your response as markdown text:
    - Grade: [text values of 'Pass' or 'Fail' only]
    - Feedback: [detailed feedback with citations from the summary. The text should be in markdown bullets and always nested under the bullet Feedback.]
    """

    # Generate the grade and feedback
    messages = [
        SystemMessage(content="You are grading a patient's understanding of a health topic."),
        HumanMessage(content=prompt)
    ]
    response = llm.invoke(messages)

    # Update the state
    state["quiz_grade"] = response.content

    # Initialize quiz_grades if it doesn't exist
    if "quiz_grades" not in state or state["quiz_grades"] is None:
        state["quiz_grades"] = []

    # Store the question and grade together in the quiz_grades list
    state["quiz_grades"].append({
        "question": question,
        "grade": response.content
    })

    # We don't add the grade to the conversation messages yet
    # It will be added in present_feedback after all questions are answered

    return state

def present_feedback(state: HealthBotState) -> HealthBotState:
    """Present the grade and feedback to the patient."""
    if not state["quiz_ready"]:
        return state

    # Check if there are more questions
    current_index = state.get("current_question_index", 0)
    num_questions = state.get("num_questions", 1)

    # If this is the last question or user has completed all questions
    if current_index >= num_questions - 1:
        # Calculate and present the total grade
        quiz_grades = state.get("quiz_grades", [])

        # Create a summary of all questions and grades
        summary = "Quiz Results:\n\n"

        # Add each question and its grade
        for i, grade_item in enumerate(quiz_grades):
            summary += f"Question {i+1}: {grade_item['question']}\n\n"

            # Parse the grade string to extract just the grade and feedback
            grade_str = grade_item['grade']

            # Check if the grade is in dictionary-like format
            if isinstance(grade_item, dict) and 'grade' in grade_item:
                # Extract just the markdown text of the grade
                summary += f"{grade_item['grade']}\n\n"
            elif grade_str.startswith('{') and "grade" in grade_str:
                try:
                    # Try to evaluate the string as a dictionary
                    import ast
                    grade_dict = ast.literal_eval(grade_str)
                    if isinstance(grade_dict, dict) and 'grade' in grade_dict:
                        # Extract just the grade value
                        summary += f"{grade_dict['grade']}\n\n"
                    else:
                        # Fallback to original string if not properly formatted
                        summary += f"{grade_str}\n\n"
                except:
                    # Fallback to original string if evaluation fails
                    summary += f"{grade_str}\n\n"
            else:
                # If not in dictionary format, use as is
                summary += f"{grade_str}\n\n"

        # Add a final summary line
        summary += f"You've completed all {num_questions} questions! Thank you for testing your knowledge."

        # Add the summary to the conversation
        state["messages"].append({
            "role": "assistant",
            "content": summary
        })

        # Print the summary
        print(f"HealthBot: {summary}")

    else:
        # Increment the question index
        state["current_question_index"] = current_index + 1

        # Ask if they want to continue to the next question
        state["messages"].append({
            "role": "assistant",
            "content": "Ready for the next question? (yes/no)"
        })

        response = input("HealthBot: Ready for the next question? (yes/no) ")

        # Add user response to the conversation
        state["messages"].append({
            "role": "user",
            "content": response
        })

        if response.lower() not in ["yes", "y", "sure", "ok", "okay"]:
            # Skip remaining questions and show the results for the questions answered
            state["current_question_index"] = num_questions

            # Recursively call present_feedback to show the results
            state = present_feedback(state)

    return state

def suggest_related_topics(state: HealthBotState) -> HealthBotState:
    """Suggest related health topics based on the current topic."""
    health_topic = state["health_topic"]
    summary = state["summary"]

    # Create a prompt for the LLM to suggest related topics
    prompt = f"""
    Based on the patient's interest in {health_topic} and the summary provided, suggest 3 related health topics that the patient might want to learn about next.

    Summary:
    {summary}

    Format your response as a JSON array of strings, with each string being a related topic.
    """

    # Generate the related topics
    messages = [
        SystemMessage(content="You are suggesting related health topics to a patient."),
        HumanMessage(content=prompt)
    ]
    response = llm.invoke(messages)

    # Parse the response to extract the related topics
    import json
    try:
        topics_text = response.content
        # Find the JSON array in the response
        start_idx = topics_text.find('[')
        end_idx = topics_text.rfind(']') + 1
        if start_idx >= 0 and end_idx > start_idx:
            topics_json = topics_text[start_idx:end_idx]
            topics = json.loads(topics_json)
        else:
            # Fallback if JSON parsing fails
            topics = ["Related topic 1", "Related topic 2", "Related topic 3"]
    except:
        # Fallback if JSON parsing fails
        topics = ["Related topic 1", "Related topic 2", "Related topic 3"]

    # Update the state
    state["related_topics"] = topics

    # Add the suggestions to the conversation
    suggestion_text = "You might also be interested in these related topics:\n"
    for i, topic in enumerate(topics):
        suggestion_text += f"{i+1}. {topic}\n"

    state["messages"].append({
        "role": "assistant",
        "content": suggestion_text
    })

    # Print the suggestions
    print(f"HealthBot: {suggestion_text}")

    return state

def ask_next_action(state: HealthBotState) -> HealthBotState:
    """Ask the patient if they'd like to learn about a new topic or exit."""
    # Add the question to the conversation
    related_topics = state.get("related_topics", [])

    if related_topics:
        prompt = "Would you like to:\n1. Learn about one of these related topics (enter the number)\n2. Learn about a new health topic (enter 'new')\n3. Exit (enter 'exit')"
    else:
        prompt = "Would you like to learn about a new health topic (enter 'new') or exit (enter 'exit')?"

    state["messages"].append({
        "role": "assistant",
        "content": prompt
    })

    # Get user input
    response = input(f"HealthBot: {prompt} ")

    # Add user response to the conversation
    state["messages"].append({
        "role": "user",
        "content": response
    })

    # Update the state
    if response.lower() in ["exit", "quit", "bye", "goodbye"]:
        state["next_action"] = "exit"
    elif response.lower() in ["new", "new topic"]:
        state["next_action"] = "new_topic"
    elif related_topics and response.isdigit() and 1 <= int(response) <= len(related_topics):
        # User selected a related topic
        selected_topic = related_topics[int(response) - 1]
        state["health_topic"] = selected_topic
        state["next_action"] = "new_topic"
    else:
        # Default to new topic
        state["next_action"] = "new_topic"

    return state

def router(state: HealthBotState) -> str:
    """Route to the next node based on the state."""
    # If the user wants to exit, end the conversation
    if state.get("next_action") == "exit":
        return "end_conversation"

    # If the user wants to learn about a new topic, restart the flow
    if state.get("next_action") == "new_topic":
        return "ask_health_topic"

    # If the user is not ready for a quiz, skip to related topics
    if state.get("quiz_ready") is False:
        return "suggest_related_topics"

    # If there are more questions to ask, go back to present_quiz_question
    current_index = state.get("current_question_index", 0)
    num_questions = state.get("num_questions", 1)
    if state.get("quiz_ready") and current_index < num_questions:
        return "present_quiz_question"

    # Otherwise, continue with the normal flow
    return "suggest_related_topics"

def end_conversation(state: HealthBotState) -> HealthBotState:
    """End the conversation with a farewell message."""
    # Add the farewell to the conversation
    state["messages"].append({
        "role": "assistant",
        "content": "Thank you for using HealthBot! Take care and stay healthy!"
    })

    # Print the farewell
    print("HealthBot: Thank you for using HealthBot! Take care and stay healthy!")

    return state

# Create the workflow
workflow = StateGraph(state_schema=HealthBotState)

# Add nodes
workflow.add_node("ask_health_topic", ask_health_topic)
workflow.add_node("search_health_info", search_health_info)
workflow.add_node("summarize_health_info", summarize_health_info)
workflow.add_node("present_summary", present_summary)
workflow.add_node("prompt_for_quiz", prompt_for_quiz)
workflow.add_node("create_quiz_questions", create_quiz_questions)
workflow.add_node("present_quiz_question", present_quiz_question)
workflow.add_node("collect_quiz_answer", collect_quiz_answer)
workflow.add_node("grade_quiz_answer", grade_quiz_answer)
workflow.add_node("present_feedback", present_feedback)
workflow.add_node("suggest_related_topics", suggest_related_topics)
workflow.add_node("ask_next_action", ask_next_action)
workflow.add_node("end_conversation", end_conversation)
workflow.add_node("router", router)

# Add edges
workflow.add_edge(START, "ask_health_topic")
workflow.add_edge("ask_health_topic", "search_health_info")
workflow.add_edge("search_health_info", "summarize_health_info")
workflow.add_edge("summarize_health_info", "present_summary")
workflow.add_edge("present_summary", "prompt_for_quiz")
workflow.add_edge("prompt_for_quiz", "create_quiz_questions")
workflow.add_edge("create_quiz_questions", "present_quiz_question")
workflow.add_edge("present_quiz_question", "collect_quiz_answer")
workflow.add_edge("collect_quiz_answer", "grade_quiz_answer")
workflow.add_edge("grade_quiz_answer", "present_feedback")
workflow.add_edge("present_feedback", "router")
workflow.add_edge("suggest_related_topics", "ask_next_action")
workflow.add_edge("ask_next_action", "router")

# Add conditional edges
workflow.add_conditional_edges(
    "router",
    {
        "ask_health_topic": lambda state: state.get("next_action") == "new_topic",
        "suggest_related_topics": lambda state: state.get("quiz_ready") is False,
        "present_quiz_question": lambda state: state.get("quiz_ready") and state.get("current_question_index", 0) < state.get("num_questions", 1),
        "end_conversation": lambda state: state.get("next_action") == "exit",
    }
)

workflow.add_edge("end_conversation", END)

# Compile the workflow
graph = workflow.compile()

# Create a Gradio interface
def healthbot_chat(message, history, difficulty="medium", level_of_detail="medium", num_questions=1):
    """Function to handle the Gradio chat interface."""
    # Initialize or get the current state
    if not hasattr(healthbot_chat, "state"):
        healthbot_chat.state = HealthBotState(
            messages=[],
            health_topic=None,
            quiz_ready=None,
            quiz_answer=None,
            next_action=None,
            difficulty=difficulty,
            level_of_details=level_of_detail,  # Set level_of_details from parameter
            num_questions=num_questions,
            current_question_index=0,
            search_results=None,
            summary=None,
            quiz_questions=None,
            current_quiz_question=None,
            quiz_grade=None,
            quiz_feedback=None,
            quiz_grades=[],
            related_topics=None
        )

    # If this is a new conversation or the user wants to restart
    if message.lower() in ["restart", "new", "new topic"]:
        healthbot_chat.state = HealthBotState(
            messages=[],
            health_topic=None,
            quiz_ready=None,
            quiz_answer=None,
            next_action=None,
            difficulty=difficulty,
            level_of_details=level_of_detail,  # Set level_of_details from parameter
            num_questions=num_questions,
            current_question_index=0,
            search_results=None,
            summary=None,
            quiz_questions=None,
            current_quiz_question=None,
            quiz_grade=None,
            quiz_feedback=None,
            quiz_grades=[],
            related_topics=None
        )
        return history + [{"role": "user", "content": message}, {"role": "assistant", "content": "What health topic or medical condition would you like to learn about today?"}], ""

    # If this is the first message, it's the health topic
    if healthbot_chat.state.get("health_topic") is None:
        healthbot_chat.state["health_topic"] = message
        healthbot_chat.state["difficulty"] = difficulty
        healthbot_chat.state["level_of_details"] = level_of_detail  # Set level_of_details from parameter
        healthbot_chat.state["num_questions"] = num_questions

        # Update the state with search results and summary
        healthbot_chat.state = search_health_info(healthbot_chat.state)
        healthbot_chat.state = summarize_health_info(healthbot_chat.state)

        # Return the summary
        return history + [{"role": "user", "content": message}, {"role": "assistant", "content": healthbot_chat.state["summary"] + "\n\nWould you like to take a quick quiz to test your understanding? (yes/no)"}], ""

    # If waiting for quiz readiness response
    if healthbot_chat.state.get("quiz_ready") is None:
        healthbot_chat.state["quiz_ready"] = message.lower() in ["yes", "y", "sure", "ok", "okay"]

        if healthbot_chat.state["quiz_ready"]:
            # Create quiz questions
            healthbot_chat.state = create_quiz_questions(healthbot_chat.state)

            # Return the first question
            return history + [{"role": "user", "content": message}, {"role": "assistant", "content": f"Question 1: {healthbot_chat.state['current_quiz_question']}"}], ""
        else:
            # Skip to related topics
            healthbot_chat.state = suggest_related_topics(healthbot_chat.state)

            # Ask for next action
            related_topics = healthbot_chat.state.get("related_topics", [])
            if related_topics:
                prompt = "Would you like to:\n1. Learn about one of these related topics (enter the number)\n2. Learn about a new health topic (enter 'new')\n3. Exit (enter 'exit')"
            else:
                prompt = "Would you like to learn about a new health topic (enter 'new') or exit (enter 'exit')?"

            suggestion_text = "You might also be interested in these related topics:\n"
            for i, topic in enumerate(related_topics):
                suggestion_text += f"{i+1}. {topic}\n"

            return history + [{"role": "user", "content": message}, {"role": "assistant", "content": suggestion_text + "\n" + prompt}], ""

    # If waiting for quiz answer
    if healthbot_chat.state.get("quiz_ready") and healthbot_chat.state.get("quiz_answer") is None:
        healthbot_chat.state["quiz_answer"] = message

        # Grade the answer
        healthbot_chat.state = grade_quiz_answer(healthbot_chat.state)

        # Check if there are more questions
        current_index = healthbot_chat.state.get("current_question_index", 0)
        num_questions = healthbot_chat.state.get("num_questions", 1)

        if current_index < num_questions - 1:
            # Increment the question index
            healthbot_chat.state["current_question_index"] = current_index + 1
            healthbot_chat.state["quiz_answer"] = None

            # Return only the next question (without showing the grade for the current question)
            next_question = healthbot_chat.state["quiz_questions"][current_index + 1]
            return history + [{"role": "user", "content": message}, {"role": "assistant", "content": f"Question {current_index + 2}: {next_question}"}], ""
        else:
            # No more questions, create a summary of all questions and grades
            quiz_grades = healthbot_chat.state.get("quiz_grades", [])
            quiz_questions = healthbot_chat.state.get("quiz_questions", [])

            # Create a summary of all questions and grades
            summary = "Quiz Results:\n\n"

            # Add each question and its grade
            for i, (question, grade_item) in enumerate(zip(quiz_questions, quiz_grades)):
                summary += f"Question {i+1}: {question}\n"

                # Parse the grade to extract just the markdown text
                if isinstance(grade_item, dict) and 'grade' in grade_item:
                    # Extract just the markdown text of the grade
                    summary += f"{grade_item['grade']}\n\n"
                elif isinstance(grade_item, str):
                    # If it's a string, check if it's in dictionary-like format
                    if grade_item.startswith('{') and "grade" in grade_item:
                        try:
                            # Try to evaluate the string as a dictionary
                            import ast
                            grade_dict = ast.literal_eval(grade_item)
                            if isinstance(grade_dict, dict) and 'grade' in grade_dict:
                                # Extract just the grade value
                                summary += f"{grade_dict['grade']}\n\n"
                            else:
                                # Fallback to original string if not properly formatted
                                summary += f"{grade_item}\n\n"
                        except:
                            # Fallback to original string if evaluation fails
                            summary += f"{grade_item}\n\n"
                    else:
                        # If not in dictionary format, use as is
                        summary += f"{grade_item}\n\n"
                else:
                    # Fallback for any other type
                    summary += f"{grade_item}\n\n"

            # Add a final summary line
            summary += f"You've completed all {num_questions} questions! Thank you for testing your knowledge.\n\n"

            # Suggest related topics
            healthbot_chat.state = suggest_related_topics(healthbot_chat.state)

            # Ask for next action
            related_topics = healthbot_chat.state.get("related_topics", [])
            if related_topics:
                prompt = "Would you like to:\n1. Learn about one of these related topics (enter the number)\n2. Learn about a new health topic (enter 'new')\n3. Exit (enter 'exit')"
            else:
                prompt = "Would you like to learn about a new health topic (enter 'new') or exit (enter 'exit')?"

            suggestion_text = "You might also be interested in these related topics:\n"
            for i, topic in enumerate(related_topics):
                suggestion_text += f"{i+1}. {topic}\n"

            return history + [{"role": "user", "content": message}, {"role": "assistant", "content": summary + suggestion_text + "\n" + prompt}], ""

    # If waiting for next action
    if healthbot_chat.state.get("related_topics") is not None and healthbot_chat.state.get("next_action") is None:
        related_topics = healthbot_chat.state.get("related_topics", [])

        if message.lower() in ["exit", "quit", "bye", "goodbye"]:
            healthbot_chat.state["next_action"] = "exit"
            return history + [{"role": "user", "content": message}, {"role": "assistant", "content": "Thank you for using HealthBot! Take care and stay healthy!"}], ""
        elif message.lower() in ["new", "new topic"]:
            # Reset the state for a new topic
            healthbot_chat.state = HealthBotState(
                messages=[],
                health_topic=None,
                quiz_ready=None,
                quiz_answer=None,
                next_action=None,
                difficulty=difficulty,
                level_of_details=level_of_detail,  # Set level_of_details from parameter
                num_questions=num_questions,
                current_question_index=0,
                search_results=None,
                summary=None,
                quiz_questions=None,
                current_quiz_question=None,
                quiz_grade=None,
                quiz_feedback=None,
                quiz_grades=[],
                related_topics=None
            )
            return history + [{"role": "user", "content": message}, {"role": "assistant", "content": "What health topic or medical condition would you like to learn about today?"}], ""
        elif related_topics and message.isdigit() and 1 <= int(message) <= len(related_topics):
            # User selected a related topic
            selected_topic = related_topics[int(message) - 1]

            # Reset the state but keep the selected topic
            healthbot_chat.state = HealthBotState(
                messages=[],
                health_topic=selected_topic,
                quiz_ready=None,
                quiz_answer=None,
                next_action=None,
                difficulty=difficulty,
                level_of_details=level_of_detail,  # Set level_of_details from parameter
                num_questions=num_questions,
                current_question_index=0,
                search_results=None,
                summary=None,
                quiz_questions=None,
                current_quiz_question=None,
                quiz_grade=None,
                quiz_feedback=None,
                quiz_grades=[],
                related_topics=None
            )

            # Update the state with search results and summary
            healthbot_chat.state = search_health_info(healthbot_chat.state)
            healthbot_chat.state = summarize_health_info(healthbot_chat.state)

            # Return the summary
            return history + [{"role": "user", "content": message}, {"role": "assistant", "content": f"Here's information about {selected_topic}:\n\n" + healthbot_chat.state["summary"] + "\n\nWould you like to take a quick quiz to test your understanding? (yes/no)"}], ""
        else:
            # Default to new topic
            healthbot_chat.state = HealthBotState(
                messages=[],
                health_topic=None,
                quiz_ready=None,
                quiz_answer=None,
                next_action=None,
                difficulty=difficulty,
                level_of_details=level_of_detail,  # Set level_of_details from parameter
                num_questions=num_questions,
                current_question_index=0,
                search_results=None,
                summary=None,
                quiz_questions=None,
                current_quiz_question=None,
                quiz_grade=None,
                quiz_feedback=None,
                quiz_grades=[],
                related_topics=None
            )
            return history + [{"role": "user", "content": message}, {"role": "assistant", "content": "What health topic or medical condition would you like to learn about today?"}], ""

    # Default response
    return history + [{"role": "user", "content": message}, {"role": "assistant", "content": "I'm not sure how to respond to that. Would you like to learn about a health topic?"}], ""


# Create the Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# HealthBot: AI-Powered Patient Education System")
    gr.Markdown("Ask about any health topic, get a summary, take a quiz, and explore related topics.")

    with gr.Row():
        with gr.Column(scale = 3):
            # Initialize chatbot with the welcome message
            initial_message = [{"role": "assistant", "content": "What health topic or medical condition would you like to learn about today?"}]
            chatbot = gr.Chatbot(height = 600, type = "messages", value=initial_message)

            # Text input (initially visible)
            msg = gr.Textbox(
                label = "Type your message here...",
                placeholder = "e.g., diabetes, asthma, heart disease",
                visible = True
            )

            # Yes/No buttons (initially hidden)
            with gr.Row():
                yes_btn = gr.Button("✅ Yes", variant = "primary", visible = False)
                no_btn = gr.Button("❌ No", variant = "secondary", visible = False)

            clear = gr.Button("Clear Conversation")

        with gr.Column(scale = 1):
            difficulty = gr.Radio(
                ["easy", "medium", "hard"],
                label = "Difficulty Level",
                info = "Select the difficulty level for quizzes",
                value = "medium"
            )

            level_of_detail = gr.Radio(
                ["easy", "medium", "hard"],
                label = "Level of Detail",
                info = "Select the level of detail for information",
                value = "medium"
            )

            num_questions = gr.Slider(
                minimum = 1,
                maximum = 5,
                step = 1,
                label = "Number of Quiz Questions",
                info = "How many questions would you like in your quiz?",
                value = 1
            )


    # Enhanced chat function that manages input/button visibility
    def enhanced_healthbot_chat(message, history, difficulty = "medium", level_of_detail = "medium", num_questions = 1):
        response_history, _ = healthbot_chat(message, history, difficulty, level_of_detail, num_questions)

        # Check if the last assistant message asks a yes/no question
        last_message = response_history[-1]["content"].lower() if response_history else ""
        is_yes_no_question = any(phrase in last_message for phrase in [
            "would you like to take a quiz",
            "ready for the next question",
            "yes/no"
        ]
                                 )

        if is_yes_no_question:
            # Hide text input, show buttons
            return (
                response_history,
                "",  # Clear text input
                gr.update(visible = False),  # Hide text input
                gr.update(visible = True),  # Show yes button
                gr.update(visible = True)  # Show no button
            )
        else:
            # Show text input, hide buttons
            return (
                response_history,
                "",  # Clear text input
                gr.update(visible = True),  # Show text input
                gr.update(visible = False),  # Hide yes button
                gr.update(visible = False)  # Hide no button
            )


    # Button click handlers
    def handle_yes_click(history, difficulty, level_of_detail, num_questions):
        # Process the "yes" response and check what to show next
        response_history, _ = healthbot_chat("yes", history, difficulty, level_of_detail, num_questions)

        # Check if the response contains another yes/no question
        last_message = response_history[-1]["content"].lower() if response_history else ""
        is_yes_no_question = any(phrase in last_message for phrase in [
            "would you like to take a quiz",
            "ready for the next question",
            "yes/no"
        ]
                                 )

        if is_yes_no_question:
            # Keep buttons visible, hide text input
            return (
                response_history,
                "",  # Clear text input
                gr.update(visible = False),  # Hide text input
                gr.update(visible = True),  # Show yes button
                gr.update(visible = True)  # Show no button
            )
        else:
            # Show text input, hide buttons
            return (
                response_history,
                "",  # Clear text input
                gr.update(visible = True),  # Show text input
                gr.update(visible = False),  # Hide yes button
                gr.update(visible = False)  # Hide no button
            )


    def handle_no_click(history, difficulty, level_of_detail, num_questions):
        # Process the "no" response and check what to show next
        response_history, _ = healthbot_chat("no", history, difficulty, level_of_detail, num_questions)

        # Check if the response contains another yes/no question
        last_message = response_history[-1]["content"].lower() if response_history else ""
        is_yes_no_question = any(phrase in last_message for phrase in [
            "would you like to take a quiz",
            "ready for the next question",
            "yes/no",
            "(yes / no)",
            "Would you like to take a quick quiz to test your understanding? (yes / no)"
        ]
                                 )

        if is_yes_no_question:
            # Keep buttons visible, hide text input
            return (
                response_history,
                "",  # Clear text input
                gr.update(visible = False),  # Hide text input
                gr.update(visible = True),  # Show yes button
                gr.update(visible = True)  # Show no button
            )
        else:
            # Show text input, hide buttons
            return (
                response_history,
                "",  # Clear text input
                gr.update(visible = True),  # Show text input
                gr.update(visible = False),  # Hide yes button
                gr.update(visible = False)  # Hide no button
            )


    # Event handlers
    msg.submit(
        enhanced_healthbot_chat,
        [msg, chatbot, difficulty, level_of_detail, num_questions],
        [chatbot, msg, msg, yes_btn, no_btn]  # Note: msg appears twice for value and visibility
    )

    yes_btn.click(
        handle_yes_click,
        [chatbot, difficulty, level_of_detail, num_questions],
        [chatbot, msg, msg, yes_btn, no_btn]  # Note: msg appears twice for value and visibility
    )

    no_btn.click(
        handle_no_click,
        [chatbot, difficulty, level_of_detail, num_questions],
        [chatbot, msg, msg, yes_btn, no_btn]  # Note: msg appears twice for value and visibility
    )

    clear.click(
        lambda: (
            None,  # Clear chatbot
            "",  # Clear text input value
            gr.update(visible = True),  # Show text input
            gr.update(visible = False),  # Hide yes button
            gr.update(visible = False)  # Hide no button
        ),
        None,
        [chatbot, msg, msg, yes_btn, no_btn],
        queue = False
    )

# Run the standalone workflow (for testing without Gradio)
def run_healthbot():
    """Run the HealthBot workflow in the terminal."""
    # Initialize the state
    state = HealthBotState(
        messages=[],
        health_topic=None,
        quiz_ready=None,
        quiz_answer=None,
        next_action=None,
        difficulty=None,
        level_of_details=None,  # Added for consistency with other state initializations
        num_questions=None,
        current_question_index=0,
        search_results=None,
        summary=None,
        quiz_questions=None,
        current_quiz_question=None,
        quiz_grade=None,
        quiz_feedback=None,
        related_topics=None
    )

    # Run the workflow
    graph.invoke(state)

# Main entry point
if __name__ == "__main__":
    # Launch the Gradio interface
    demo.launch()

    # Alternatively, run the terminal version
    # run_healthbot()
