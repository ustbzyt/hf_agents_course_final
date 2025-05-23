import logging
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from .agent_state import AgentState
from .utils import agent_runnable
import time

def assistant(state: AgentState) -> AgentState:
    if not state.get('messages'):
        tools_desc = (
            "duckduckgo_search(query: str) -> str: Performs a web search using DuckDuckGo to find information on the internet.\n"
            "wikipedia_search(query: str) -> str: Searches Wikipedia for up-to-date encyclopedic information.\n"
            "tavily_search(query: str) -> str: Performs a web search using Tavily API to find up-to-date information from the internet.\n"
            "serpapi_search(query: str) -> str: Performs a web search using SerpAPI to retrieve information from Google and other sources.\n"
            "requests_get(url: str) -> str: Fetches the main content of a web page by URL.\n"
            "youtube_search(query: str) -> str: Searches YouTube for videos related to the query.\n"
        )
        sys_prompt = (
            "You are a general AI assistant. I will ask you a question.\n"
            "\n"
            "When you receive a question, follow these steps:\n"
            "1. Think step by step and write your plan.\n"
            "3. Execute your plan, calling tools as needed.\n"
            "4. After each tool response, ALWAYS write your reflection on the result.\n"
            "5. NEVER reply with an empty message.\n"
            "6. Repeat tool calls as needed until you have enough information.\n"
            "7. If confident, output your FINAL ANSWER in the required format. If you cannot answer after reasonable attempts, state that the answer cannot be found and stop.\n"
            "\n"
            "Rules for your FINAL ANSWER:\n"
            "- If a number is required, do not use commas, units (like $ or %), unless specified.\n"
            "- If a string is required, do not use articles, do not abbreviate (e.g. for cities), and write digits in plain text unless specified.\n"
            "- If a comma separated list is required, apply the above rules to each element.\n"
            "\n"
            "You MUST use the tools below to answer. Do NOT answer directly without using tools.\n"
            "\n"
            "The available tools are:\n"
            f"{tools_desc}"
            "- Call tools by function name and parameters.\n"
            "- You may chain or loop tool calls as needed.\n"
            "\n"
            "Keep your output clear and concise. Your final answer must start with: FINAL ANSWER: ...\n"
        )
        state["messages"] = [SystemMessage(content=sys_prompt)]
        if "question" in state and state["question"]:
            state["messages"].append(HumanMessage(content=state["question"]))
    if state["messages"]:
        last_msg = state["messages"][-1]
        if isinstance(last_msg, HumanMessage):
            logging.debug(f"Latest HumanMessage: {last_msg.content}")
        elif isinstance(last_msg, SystemMessage):
            logging.debug("System prompt sent.")
    result = agent_runnable.invoke(state["messages"])
    time.sleep(4)
    if isinstance(result, AIMessage):
        msg_type = getattr(result, "type", "AIMessage")
        content = getattr(result, "content", "No content")
        if not content or not str(content).strip():
            logging.debug(f"Gemini produced an empty response. Raw reply: {result}")
            result.content = "No content"
        logging.debug(f"Gemini {msg_type} reply: {content}")
    else:
        logging.debug(f"Gemini reply (non-AIMessage): {result}")
    if isinstance(result, AIMessage):
        if "final answer" in result.content.lower():
            result.type = "final"
            content = result.content
            idx = content.lower().find("final answer:")
            if idx != -1:
                final = content[idx + len("final answer:"):].strip()
                final = final.replace("\n", " ").strip()
                state["final_answer"] = final
            else:
                state["final_answer"] = content.strip()
        else:
            result.type = "intermediate"
            state["final_answer"] = None
    state["messages"].append(result)
    return state