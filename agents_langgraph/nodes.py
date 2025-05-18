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
            "When you receive a question:\n"
            "1. Think step by step and write out your chain of thought.\n"
            "2. Write an execution plan (list the tool calls you plan to make and why).\n"
            "3. Execute your plan, calling tools as needed.\n"
            "4. After each tool response, reflect: did you get what you need? If not, replan and continue.\n"
            "5. You may call tools multiple times until you have enough information.\n"
            "6. When you are confident, output your FINAL ANSWER in the required format.\n"
            "\n"
            "Rules for your FINAL ANSWER:\n"
            "- If a number is required, do not use commas, units (like $ or %), unless specified.\n"
            "- If a string is required, do not use articles, do not abbreviate (e.g. for cities), and write digits in plain text unless specified.\n"
            "- If a comma separated list is required, apply the above rules to each element.\n"
            "\n"
            "You MUST use the tools provided below to answer my question.\n"
            "Do NOT attempt to answer the question directly without using the tools.\n"
            "\n"
            "The available tools are:\n"
            f"{tools_desc}"
            "- You can call them by using the function name and passing the required parameters.\n"
            "- You can chain tools: call one tool, use its output as input for another.\n"
            "- You can loop: call a tool, use its output to call another, and so on until you get the final answer.\n"
            "\n"
            "Always follow the above steps. Keep your output clear, structured, and concise.\n"
            "Your final answer must start with: FINAL ANSWER: ...\n"
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
        logging.debug(f"Gemini {msg_type} reply: {getattr(result, 'content', result)}")
    else:
        logging.debug(f"Gemini reply: {getattr(result, 'content', result)}")
    if not getattr(result, "content", None) or not str(result.content).strip():
        logging.debug(f"Gemini produced an empty response. Raw reply: {result}")
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