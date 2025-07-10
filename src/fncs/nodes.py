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

def search_health_info(state: HealthBotState) -> HealthBotState:
    """
    Searches for health-related information based on the provided state and updates the state to include the
    search results. The search query is constructed dynamically depending on the specified level of details
    (easy, medium, hard) and the given health topic.

    :param state: A dictionary-like object representing the current state, which includes the "health_topic"
        indicating the topic to search for and optionally the "level_of_details" or "difficulty" to determine
        how detailed the returned information should be.
    :type state: HealthBotState
    :return: Updated state with the search results added under the "search_results" key.
    :rtype: HealthBotState
    """
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
    """
    Summarizes health information about a specific topic based on the search results
    and updates the interaction state.

    The function generates a patient-friendly summary using an AI language model.
    The complexity and detail level of the summary are adjusted based on the
    `level_of_details` (or `difficulty`) value in the state. The final summary is
    stored in the `state`, and the conversation history is updated with the
    assistant's response.

    :param state: A dictionary representing the HealthBotState. It includes search
        results, the health topic to be summarized, the required detail level, and
        a history of conversation messages.
    :type state: HealthBotState

    :return: The updated state with the newly generated summary and a record of the
        assistant's response in the message history.
    :rtype: HealthBotState
    """
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
    """
    Prompts the user for a quiz to assess their understanding and updates the state accordingly.

    The function interacts with the user by updating the conversation history in the
    state with messages from both the assistant and the user. If the user agrees to
    take the quiz, the function requests the desired number of questions, validates
    the input, and updates the state with the number of questions to ask. If the input
    is invalid, a default value of 1 is used.

    :param state: The current state of the HealthBot, which includes a conversation
        history and other context information.
    :type state: HealthBotState
    :return: Updated HealthBotState after interacting with the user and setting quiz-
        related attributes.
    :rtype: HealthBotState
    """
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
    """
    Generates quiz questions based on a provided summary within the state. The function
    takes into account the health topic, specified difficulty, and number of desired
    questions to create a series of quiz questions that test a patient's understanding
    of the health-related material.

    :param state: The current state of the health bot, containing relevant keys such as
        "quiz_ready," "summary," "health_topic," "difficulty," and "num_questions."
        Modifications to this state structure are made directly within the function.
        Expected to be a dictionary-like structure based on `HealthBotState` type hint.
    :return: An updated `HealthBotState` object containing the generated quiz questions.
    """
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
    """
    Presents the next quiz question to the user if the quiz is ready. This function updates the
    conversation messages with the current quiz question and prints it to the console. The
    state is updated with the current quiz question.

    :param state: A mutable dictionary representing the current state of the HealthBot,
        including readiness of the quiz, list of questions, conversation history, and
        the current question index. The dictionary should contain the following keys:
            - 'quiz_ready' (bool): Indicates if the quiz is ready to be presented.
            - 'quiz_questions' (list[str]): A list of questions for the quiz.
            - 'messages' (list[dict]): The conversation history, where each dictionary has
              a 'role' and 'content'.
            - 'current_question_index' (Optional[int]): The index of the current question.
            - 'current_quiz_question' (Optional[str]): The content of the current quiz question.

    :return: The updated state after presenting the next quiz question. The updated state
        includes the question being added to the conversation messages and the current quiz
        question being updated.
    :rtype: HealthBotState
    """
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
    """
    Collects a user's quiz answer by prompting for input, updates the state with
    the received answer, and appends it as a user message in the conversation history.

    :param state: A dictionary representing the current bot's state with key-value
                  pairs like "quiz_ready", "messages", and "quiz_answer".
    :type state: HealthBotState

    :return: The updated state after collecting and storing the user's quiz answer.
    :rtype: HealthBotState
    """
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
    """
    Grades a patient's quiz answer to a health-related question, updates the
    state with grading results, and provides detailed feedback using
    information from a summary.

    :param state: A dictionary representing the current state of the quiz,
        including the current quiz question, the patient's answer, and a
        summary of information relevant to grading the answer.
    :type state: HealthBotState
    :return: Updated state with the grading outcome and feedback for the
        patient's quiz answer. It also includes a record of the question and
        its grade in the quiz_grades list.
    :rtype: HealthBotState
    """
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
    """
    Processes the current quiz state and presents feedback to the user, either after
    each question or at the end of the quiz. Updates the conversation state and
    handles user interactions such as continuing to the next question or completing
    the quiz. Provides a summary of quiz results and feedback for each question.

    :param state: The current state of the quiz, represented as a dictionary
        containing quiz settings, progress, messages, grades, and other relevant data.
        Keys typically include:
        - "quiz_ready" (bool): Indicates if the quiz is ready to start.
        - "current_question_index" (int): Tracks the index of the current question.
        - "num_questions" (int): Total number of questions in the quiz.
        - "quiz_grades" (list of dict): Stores feedback and grades for each question.
        - "messages" (list of dict): Holds the conversation history between the bot
          and the user, where each message is a dictionary with keys "role" and "content".

    :return: The updated state dictionary reflecting any changes made during the
        function execution, such as incremented question index, updated conversation
        messages, or quiz completion summary.
    """
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
    """
    Suggests related health topics based on the patient's interest and a provided summary, using a
    language model to generate suggestions and appending the results to the system's state.

    :param state: A dictionary representing the current state of the Health Bot, containing the keys
        - "health_topic" (str): The health topic of interest to the patient.
        - "summary" (str): A summary to provide additional context for suggesting related topics.
        - "messages" (list[dict]): A list of conversation messages where suggestions will be appended.
    :return: The updated state dictionary, augmented with
        - "related_topics" (list[str]): A list of suggested related health topics.
    """
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
    """
    Asks the user for the next action to take based on the current state of the conversation.

    This function constructs a prompt for the user, incorporating related health topics
    (if available) into the dialogue. It subsequently accepts user input, updates the state
    based on the input, and determines the next course of action for HealthBot.

    :param state: A dictionary-like object that represents the current state of HealthBot.
        It includes keys such as `related_topics` (a list of potential health topics to
        explore), `messages` (a list of dictionary objects representing the conversation history),
        and `next_action` (a string representing the determined next step).
    :type state: HealthBotState

    :return: The updated state of HealthBot, with the user input and the next action determined
        based on the user's response.
    :rtype: HealthBotState
    """
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
    """
    Routes the application logic based on the current state.

    This function takes in the current state of the application, evaluates the
    user's intent or situation based on the state attributes, and returns the
    next stage or step in the application's flow. It supports multiple scenarios
    such as exiting the application, restarting flows, skipping quizzes, and
    continuing the quiz process. The decision-making is dependent on specific
    state attributes and their values.

    :param state: The current state of the application containing attributes
        specifying the user's context and actions.
        - "next_action": A string indicating the user's immediate intent
          (e.g., "exit", "new_topic").
        - "quiz_ready": A boolean flag indicating if the user is ready to
          participate in a quiz.
        - "current_question_index": An integer indicating the index of the
          current quiz question (default 0).
        - "num_questions": An integer representing the total number of questions
          in the quiz (default 1).
    :type state: HealthBotState

    :return: A string representing the next action to take, which corresponds
        to flow control in the application.
    :rtype: str
    """
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


######### ######### ######### ######### #########


######### Create the workflow as a function by using the node functions and the router function. #########
def wrk_flow() -> HealthBotState:

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
    return graph

######### ######### ######### ######### #########
