import os
from typing import Any, List
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import Runnable
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain_community.tools.youtube.search import YouTubeSearchTool

duckduckgo_search = DuckDuckGoSearchRun()
wikipedia_search = load_tools(["wikipedia"])
tavily_search = TavilySearchResults(api_key=os.getenv("TAVILY_API_KEY"))
serpapi_search = load_tools(["serpapi"])
requests_get = load_tools(["requests_all"], allow_dangerous_tools=True)
youtube_search = YouTubeSearchTool()

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
    print(f"Error initializing Gemini model or binding tools: {e}")
    raise RuntimeError(f"Failed to create Gemini agent runnable: {e}") from e
