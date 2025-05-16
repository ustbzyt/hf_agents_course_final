import pandas as pd
from agents_langgraph.langfuse_client import langfuse_handler
from agents_langgraph.agent_core import react_graph
from langchain_core.messages import HumanMessage, AIMessage
import os
import requests
import gradio as gr
import time

default_api_url = "https://agents-course-unit4-scoring.hf.space"

class BasicAgent:
    def __init__(self):
        print("BasicAgent initialized.")
    def __call__(self, question: str) -> str:
        if not question or not question.strip():
            print("Received empty question, skipping.")
            return ""
        result = react_graph.invoke(
            input={"messages": [], "question": question},
            config={"callbacks": [langfuse_handler]}
        )
        messages = result["messages"]
        while True:
            if isinstance(messages[-1], AIMessage) and getattr(messages[-1], "type", None) == "final":
                answer = result.get("final_answer", messages[-1].content)
                print(f"Agent returning answer: {answer}")
                return answer
            result = react_graph.invoke(
                input={"messages": [], "question": question},
                config={"callbacks": [langfuse_handler]}
            )
            messages = result["messages"]

def run_and_submit_all(profile: gr.OAuthProfile | None):
    space_id = os.getenv("SPACE_ID")
    if profile:
        username = f"{profile.username}"
        print(f"User logged in: {username}")
    else:
        print("User not logged in.")
        return "Please Login to Hugging Face with the button.", None

    api_url = default_api_url
    questions_url = f"{api_url}/questions"
    submit_url = f"{api_url}/submit"

    try:
        agent = BasicAgent()
    except Exception as e:
        print(f"Error instantiating agent: {e}")
        return f"Error initializing agent: {e}", None
    agent_code = f"https://huggingface.co/spaces/{space_id}/tree/main"
    print(agent_code)

    print(f"Fetching questions from: {questions_url}")
    try:
        response = requests.get(questions_url, timeout=15)
        response.raise_for_status()
        questions_data = response.json()
        if not questions_data:
            print("Fetched questions list is empty.")
            return "Fetched questions list is empty or invalid format.", None
        print(f"Fetched {len(questions_data)} questions.")
    except requests.exceptions.RequestException as e:
        print(f"Error fetching questions: {e}")
        return f"Error fetching questions: {e}", None
    except requests.exceptions.JSONDecodeError as e:
        print(f"Error decoding JSON response from questions endpoint: {e}")
        print(f"Response text: {response.text[:500]}")
        return f"Error decoding server response for questions: {e}", None
    except Exception as e:
        print(f"An unexpected error occurred fetching questions: {e}")
        return f"An unexpected error occurred fetching questions: {e}", None

    results_log = []
    answers_payload = []
    print(f"Running agent on {len(questions_data)} questions...")
    for item in questions_data:
        task_id = item.get("task_id")
        question_text = item.get("question")
        if not task_id or question_text is None:
            continue
        try:
            answer = agent(question_text)
            answers_payload.append({"task_id": task_id, "submitted_answer": answer})
            results_log.append({"Task ID": task_id, "Question": question_text, "Submitted Answer": answer})
        except Exception as e:
            print(f"Error answering question {task_id}: {e}")
            answers_payload.append({"task_id": task_id, "submitted_answer": f"Error: {e}"})
            results_log.append({"Task ID": task_id, "Question": question_text, "Submitted Answer": f"Error: {e}"})
        time.sleep(3)

    if not answers_payload:
        return "No answers generated.", pd.DataFrame(results_log)

    submission_data = {"username": username.strip(), "agent_code": agent_code, "answers": answers_payload}
    print(f"Submitting {len(answers_payload)} answers to: {submit_url}")
    try:
        submit_response = requests.post(submit_url, json=submission_data, timeout=30)
        submit_response.raise_for_status()
        result = submit_response.json()
        final_status = (
            f"Submission Successful!\n"
            f"User: {result.get('username')}\n"
            f"Overall Score: {result.get('score', 'N/A')}% "
            f"({result.get('correct_count', '?')}/{result.get('total_attempted', '?')} correct)\n"
            f"Message: {result.get('message', 'No message received.')}"
        )
        print(f"Submission result: {result}")
        return final_status, pd.DataFrame(results_log)
    except requests.exceptions.HTTPError as e:
        print(f"HTTP error during submission: {e}")
        status_message = f"Submission Failed: Server responded with status {e.response.status_code}."
        try:
            error_json = e.response.json()
            status_message += f" Detail: {error_json.get('detail', e.response.text)}"
        except requests.exceptions.JSONDecodeError:
            status_message += f" Response: {e.response.text[:500]}"
        return status_message, pd.DataFrame(results_log)
    except requests.exceptions.Timeout:
        print("Submission request timed out.")
        return "Submission Failed: The request timed out.", pd.DataFrame(results_log)
    except requests.exceptions.RequestException as e:
        print(f"Submission request error: {e}")
        return f"Submission Failed: Network error - {e}", pd.DataFrame(results_log)
    except Exception as e:
        print(f"Unexpected error during submission: {e}")
        return f"An unexpected error occurred during submission: {e}", pd.DataFrame(results_log)

with gr.Blocks() as demo:
    gr.Markdown("# Basic Agent Evaluation Runner")
    gr.Markdown(
        """
        **Instructions:**
        1.  Please clone this space, then modify the code to define your agent's logic, the tools, the necessary packages, etc ...
        2.  Log in to your Hugging Face account using the button below. This uses your HF username for submission.
        3.  Click 'Run Evaluation & Submit All Answers' to fetch questions, run your agent, submit answers, and see the score.
        ---
        Once clicking on the "submit" button, it can take quite some time (this is the time for the agent to go through all the questions).
        """
    )
    gr.LoginButton()
    run_button = gr.Button("Run Evaluation & Submit All Answers")
    status_output = gr.Textbox(label="Run Status / Submission Result", lines=5, interactive=False)
    results_table = gr.DataFrame(label="Questions and Agent Answers", wrap=True)
    run_button.click(
        fn=run_and_submit_all,
        outputs=[status_output, results_table]
    )

if __name__ == "__main__":
    print("\n" + "-"*30 + " App Starting " + "-"*30)
    space_host_startup = os.getenv("SPACE_HOST")
    space_id_startup = os.getenv("SPACE_ID")
    if space_host_startup:
        print(f"SPACE_HOST: {space_host_startup}")
    if space_id_startup:
        print(f"SPACE_ID: {space_id_startup}")
    print("-"*(60 + len(" App Starting ")) + "\n")
    print("Launching Gradio Interface for Basic Agent Evaluation...")
    demo.launch(debug=True, share=False)