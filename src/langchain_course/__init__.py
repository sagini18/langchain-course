from typing import List
from pydantic import BaseModel, Field

from dotenv import load_dotenv
load_dotenv()

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain.messages import HumanMessage
from langchain_tavily import TavilySearch
from langchain.agents.structured_output import ToolStrategy

class Source(BaseModel):
    """Schema for a source used by an agent"""
    url:str = Field(description="The URL of the source")

class AgentResponse(BaseModel):
    """Schema for agent response with answer and sources"""
    answer:str = Field(description="The agent's answer to the query")
    sources:List[Source] = Field(default_factory=list, description="list of sources used to generate the answer")

tools = [TavilySearch()]
llm = ChatOpenAI(model="gpt-4o")
agent = create_agent(model=llm, tools=tools, response_format=AgentResponse)

def main() -> None:
    print("Hello from langchain-course!")
    response = agent.invoke({"messages":[HumanMessage(content="Search for 3 job postings for an ai engineer using langchain in the bay area on linkedin and list their details")]})
    print(response)

if __name__ == "__main__":
    main()