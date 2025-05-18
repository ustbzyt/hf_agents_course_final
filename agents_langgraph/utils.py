import os
import logging
from typing import Any, List
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import Runnable
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain_community.tools.youtube.search import YouTubeSearchTool

class LoggingDuckDuckGoSearchRun(DuckDuckGoSearchRun):
    def __call__(self, *args, **kwargs):
        logging.info(f"[TOOL] duckduckgo_search called with args={args}, kwargs={kwargs}")
        result = super().__call__(*args, **kwargs)
        logging.info(f"[TOOL] duckduckgo_search result: {result}")
        return result

class LoggingTavilySearchResults(TavilySearchResults):
    def __call__(self, *args, **kwargs):
        logging.info(f"[TOOL] tavily_search called with args={args}, kwargs={kwargs}")
        result = super().__call__(*args, **kwargs)
        logging.info(f"[TOOL] tavily_search result: {result}")
        return result

class LoggingYouTubeSearchTool(YouTubeSearchTool):
    def __call__(self, *args, **kwargs):
        logging.info(f"[TOOL] youtube_search called with args={args}, kwargs={kwargs}")
        result = super().__call__(*args, **kwargs)
        logging.info(f"[TOOL] youtube_search result: {result}")
        return result

duckduckgo_search = LoggingDuckDuckGoSearchRun()
tavily_search = LoggingTavilySearchResults(api_key=os.getenv("TAVILY_API_KEY"))
youtube_search = LoggingYouTubeSearchTool()

def log_tool_wrapper(tool, name=None):
    def wrapper(*args, **kwargs):
        logging.info(f"[TOOL] {name or getattr(tool, 'name', repr(tool))} called with args={args}, kwargs={kwargs}")
        result = tool(*args, **kwargs)
        logging.info(f"[TOOL] {name or getattr(tool, 'name', repr(tool))} result: {result}")
        return result
    wrapper.__name__ = getattr(tool, '__name__', name or repr(tool))
    wrapper.__doc__ = getattr(tool, '__doc__', None) or f"Tool wrapper for {name or repr(tool)}."
    return wrapper

def log_tool_func_wrapper(tool, name=None):
    """Wraps a tool's func to add logging, preserving signature and docstring."""
    import functools
    func = getattr(tool, 'func', None)
    if func is None:
        return tool  # Not a Tool instance or no func, skip
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logging.info(f"[TOOL] {name or getattr(tool, 'name', repr(tool))} called with args={args}, kwargs={kwargs}")
        result = func(*args, **kwargs)
        logging.info(f"[TOOL] {name or getattr(tool, 'name', repr(tool))} result: {result}")
        return result
    tool.func = wrapper
    return tool

wikipedia_search = [log_tool_func_wrapper(t, name=getattr(t, 'name', 'wikipedia_search')) for t in load_tools(["wikipedia"])]
serpapi_search = [log_tool_func_wrapper(t, name=getattr(t, 'name', 'serpapi_search')) for t in load_tools(["serpapi"])]
requests_get = [log_tool_func_wrapper(t, name=getattr(t, 'name', 'requests_get')) for t in load_tools(["requests_all"], allow_dangerous_tools=True)]

tools: List[Any] = [
    duckduckgo_search,
    tavily_search,
    youtube_search
] + wikipedia_search + serpapi_search + requests_get

try:
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set.")
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=gemini_api_key
    )
    llm_with_tools = llm.bind_tools(tools)
    agent_runnable: Runnable = llm_with_tools
except Exception as e:
    logging.error(f"Error initializing Gemini model or binding tools: {e}")
    raise RuntimeError(f"Failed to create Gemini agent runnable: {e}") from e
