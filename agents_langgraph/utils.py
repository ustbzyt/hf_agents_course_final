import os
import requests
from langchain_core.tools import tool
from dataclasses import dataclass
from typing import Dict, Any, List
from .retriever import guest_info_retriever
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import Runnable
from langchain_community.tools import DuckDuckGoSearchRun, WikipediaQueryRun, ArxivQueryRun
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain_community.tools.requests.tool import RequestsGetTool
from langchain_community.tools.youtube.search import YouTubeSearchTool

@dataclass
class WeatherData:
    main_weather: str
    description: str
    temp: float
    feels_like: float
    humidity: int
    wind_speed: float
    city_name: str

def _parse_weather_data(data: Dict[str, Any]) -> WeatherData:
    weather_info = data.get('weather', [{}])[0]
    main_info = data.get('main', {})
    wind_info = data.get('wind', {})
    return WeatherData(
        main_weather=weather_info.get('main', 'N/A'),
        description=weather_info.get('description', 'N/A'),
        temp=main_info.get('temp'),
        feels_like=main_info.get('feels_like'),
        humidity=main_info.get('humidity'),
        wind_speed=wind_info.get('speed'),
        city_name=data.get('name', 'Unknown')
    )

def _format_weather_output(weather: WeatherData) -> str:
    temp_str = f"{weather.temp}°C" if weather.temp is not None else "N/A"
    feels_like_str = f"{weather.feels_like}°C" if weather.feels_like is not None else "N/A"
    humidity_str = f"{weather.humidity}%" if weather.humidity is not None else "N/A"
    wind_speed_str = f"{weather.wind_speed} m/s" if weather.wind_speed is not None else "N/A"
    return (
        f"Weather in {weather.city_name}:\n"
        f"- Condition: {weather.main_weather} ({weather.description})\n"
        f"- Temperature: {temp_str} (Feels like: {feels_like_str})\n"
        f"- Humidity: {humidity_str}\n"
        f"- Wind Speed: {wind_speed_str}"
    )

@tool
def get_weather_info(location: str) -> str:
    """Fetches real-time weather information for a given location using OpenWeatherMap API."""
    api_key = os.getenv("OPENWEATHERMAP_API_KEY")
    if not api_key:
        return "Error: Weather API key not configured."
    base_url = "http://api.openweathermap.org/data/2.5/weather"
    params = {
        'q': location,
        'appid': api_key,
        'units': 'metric'
    }
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()
        if data.get("cod") != 200:
            error_message = data.get("message", "Unknown API error")
            return f"Error fetching weather for '{location}': {error_message}"
        weather = _parse_weather_data(data)
        return _format_weather_output(weather)
    except requests.exceptions.HTTPError as http_err:
        status_code = http_err.response.status_code
        if status_code == 401:
            return f"Error fetching weather for '{location}': Invalid API key or subscription issue."
        elif status_code == 404:
            return f"Error fetching weather: Location '{location}' not found."
        else:
            return f"HTTP error occurred while fetching weather for '{location}': {http_err}"
    except requests.exceptions.RequestException as req_err:
        return f"Error connecting to weather service for '{location}': {req_err}"
    except Exception as e:
        return f"An unexpected error occurred while processing weather for '{location}'."

search_tool = DuckDuckGoSearchRun()
wikipedia_tools = load_tools(["wikipedia"])
arxiv_tool = ArxivQueryRun()
tavily_tool = TavilySearchResults(api_key=os.getenv("TAVILY_API_KEY"))
serpapi_tools = load_tools(["serpapi"])
requests_tools = load_tools(["requests_all"], allow_dangerous_tools=True)
youtube_search_tool = YouTubeSearchTool()

# 合并 wikipedia_tools、serpapi_tools、requests_tools

tools: List[Any] = [
    guest_info_retriever,
    get_weather_info,
    search_tool,
    arxiv_tool,
    tavily_tool,
    youtube_search_tool,
] + serpapi_tools + wikipedia_tools + requests_tools

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
