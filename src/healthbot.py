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


##### ##### ##### ##### ##### ##### ##### ##### ##### ##
##### Creating the workflow from nodes and routers #####


from src.fncs.nodes import *

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



###### ###### ###### #######
###### ###### ###### #######
###### ###### ###### #######
###### ###### ###### #######
###### ###### ###### #######
###### Gradio UI/APP  ######

import gradio as gr

# Create a Gradio interface
def healthbot_chat(message, history, difficulty="medium", level_of_detail="medium", num_questions=1):
    """
    This function simulates an interactive chat with a health-related bot that provides
    information on health topics, quizzes on the user's understanding, and suggests
    related topics or actions based on user inputs. The conversation is stateful,
    persisting data such as health topics, quiz questions, answers, and feedback
    across interactions.

    :param message: The user's current message/input to the bot.
    :type message: str
    :param history: The history of the conversation, with alternating user and bot messages.
    :type history: list[dict]
    :param difficulty: The difficulty level for quiz questions ("easy", "medium", "hard").
                       Defaults to "medium".
    :type difficulty: str, optional
    :param level_of_detail: The level of detail required in health explanations ("low",
                            "medium", "high"). Defaults to "medium".
    :type level_of_detail: str, optional
    :param num_questions: The number of quiz questions to generate. Defaults to 1.
    :type num_questions: int, optional
    :return: A tuple of the updated conversation history (including the bot's response)
             and a string representing additional context or metadata.
    :rtype: tuple[list[dict], str]
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
with gr.Blocks() as app:
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
        """
        Enhances interaction with a healthbot by managing the state of input/output visibility
        based on the type of message received. It integrates conversational history and adjusts
        the interaction context dynamically. This function is particularly designed to determine
        if a yes/no question was asked and modifies the interface elements accordingly.

        :param message: Input message from the user.
        :type message: str
        :param history: The conversational history between the user and the healthbot.
        :type history: list[dict]
        :param difficulty: Difficulty level of interaction, default is "medium".
        :type difficulty: str, optional
        :param level_of_detail: Level of detail in the responses, default is "medium".
        :type level_of_detail: str, optional
        :param num_questions: Number of quiz questions to be generated, default is 1.
        :type num_questions: int, optional
        :return: A tuple containing updated response history, cleared text input,
                 visibility updates for text input, and yes/no buttons.
        :rtype: tuple
        """
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
        """
        Processes a "yes" response in the conversation and determines the next steps based on
        the response content. The function checks if the last message in the response history
        contains a yes/no question to decide whether to display buttons or a text input to
        the user.

        :param history: List of dictionaries containing the conversation history.
        :param difficulty: String indicating the difficulty level of the conversation or quiz.
        :param level_of_detail: String specifying the detail level in responses or content.
        :param num_questions: Integer representing the number of questions or steps processed.
        :return: Tuple containing updated conversation history, cleared text input, visibility
                 updates for text input, "yes" button, and "no" button.
        """
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
        """
        Processes a "no" response in an interactive dialogue or quiz scenario and determines what
        UI elements should be displayed next based on the type of question that follows. This function
        relies on the `healthbot_chat` method to generate a response and checks whether the
        follow-up message is in a yes/no question format. It accordingly updates visibility for text
        inputs or buttons.

        :param history: The dialogue history between the user and the bot.
        :type history: list
        :param difficulty: The current difficulty level for quiz or dialogue processing.
        :type difficulty: str
        :param level_of_detail: The level of detail expected in responses or questions.
        :type level_of_detail: str
        :param num_questions: The number of questions asked or required in a session.
        :type num_questions: int
        :return: A tuple containing the updated dialogue history, cleared input,
                 and visibility settings for text input and buttons.
        :rtype: tuple
        """
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
    """
    Runs the HealthBot workflow.

    This function initializes the `HealthBotState` object with its default values
    to manage the state throughout the workflow execution. It then invokes the
    workflow mechanism using the initialized state. The `HealthBotState` tracks
    various aspects during the bot's operation, including messages, quiz
    parameters, health topics, search results, and other key details required
    for managing the conversation or health-related tasks.

    :return: None
    """
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
    app.launch()

    # Alternatively, run the terminal version as shown below:
    # run_healthbot()