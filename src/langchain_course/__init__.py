from dotenv import load_dotenv

load_dotenv()

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from langchain_tavily import TavilySearch

search_tool = TavilySearch(max_results=3)

@tool
def search(query: str) -> str:
    """
    Tool that searches over the internet

    Args:
        query: The query to search for 
    Result:
        The search result
    """
    print(f"Searching for {query}")
    return search_tool.run(query) # or search_tool.invoke({"query":query})

llm = ChatOllama(model="llama3.2:3b")
tools = [search]
agent = create_agent(model=llm, tools=tools)

def main() -> None:
    print("Hello from langchain-course!")
    result_weather = agent.invoke({"messages": [HumanMessage(content="What's the weather in Tokyo?")]})
    print(result_weather)

    result_job_roles = agent.invoke({"messages": [HumanMessage(content="search for 3 job postings for an ai engineer using langchain in the bay area on linkedin and list their details")]})
    print(result_job_roles)

if __name__ == "__main__":
    main()

    # 'The current weather in Tokyo is overcast with a temperature of 12.5°C (54.5°F). '
    # 'There is a gentle breeze blowing at 6.7 km/h (ENE direction) and a pressure of 1019.0 hPa. '
    # 'The humidity is 54% and the cloud cover is 100%.'
    # ''
    # ' It feels like 9.7°C (49.5°F) due to the wind chill, and the dew point is 3.5°C (38.2°F). '
    # 'There is no rain or snow in the forecast.'



    # 'Here are the details of three job postings for an AI Engineer using Langchain in the Bay Area on LinkedIn:\n\n'
    # '1. **Chime**\n\t* Job Title: AI Engineer\n\t* Location: San Francisco Bay Area\n\t'
    # '* Salary: Not specified\n\t* Description: Actively hiring AI Engineer with experience in Langchain and other AI/ML frameworks.\n'
    # '2. **CURE Auto Insurance (Citizens United Reciprocal Exchange)**\n\t* Job Title: AI Engineer\n\t* Location: Southfield, MI\n\t'
    # '* Salary: Not specified\n\t* Description: Actively hiring AI Engineer with experience in Langchain and other AI/ML frameworks.\n'
    # '3. **7th Sky Technologies LLC**\n\t* Job Title: AI/ML Engineer\n\t* Location: Palo Alto, CA\n\t* Salary: $180,000.00 - $220,000.00 per year\n\t'
    # '* Description: Actively hiring AI/ML Engineer with experience in Langchain, LangGraph, Semantic Kernel, or similar frameworks.'