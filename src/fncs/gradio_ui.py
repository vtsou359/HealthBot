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
# Create a Gradio interface
def healthbot_chat(message, history, difficulty="medium", level_of_detail="medium", num_questions=1):
    """
    Initializes or manages a conversation with a health chatbot that provides information about health topics,
    facilitates quizzes, and suggests related topics based on user input. The chatbot dynamically responds to
    user messages by maintaining a conversational state and offers a seamless user interaction experience.

    :param message: The input message from the user as a string, directing the conversation flow (e.g.,
                    topic inquiry, quiz responses).
    :type message: str
    :param history: The list of historical messages exchanged between the user and the assistant during the
                    conversation. Each element in the list is a dictionary with "role" and "content" keys.
    :type history: list[dict[str, str]]
    :param difficulty: The difficulty level for health information or quiz questions. Defaults to "medium".
    :type difficulty: str, optional
    :param level_of_detail: The desired amount of detail in responses ("low", "medium", "high").
                            Defaults to "medium".
    :type level_of_detail: str, optional
    :param num_questions: The number of quiz questions to generate if the user opts for a quiz. Defaults to 1.
    :type num_questions: int, optional
    :return: A tuple containing the updated conversation history and an empty string. The history includes
             new entries based on the user's input and chatbot's response.
    :rtype: tuple[list[dict[str, str]], str]
    """
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
