from dotenv import load_dotenv
load_dotenv()  # take environment variables from .env.
from langchain_groq import ChatGroq
from langchain.agents import create_agent
from langchain.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langchain.tools import tool
from typing import Dict, Any
import sys
from tavily import TavilyClient

tavily_client = TavilyClient()

system_prompt = """

You are a personal chef. The user will give you a list of ingredients they have left over in their house.

Using the web search tool, search the web for recipes that can be made with the ingredients they have.

Return recipe suggestions and eventually the recipe instructions to the user, if requested.

"""

@tool
def web_search(query: str) -> Dict[str, Any]:
    """
    Search the web for recipes that can be made with the ingredients provided by the user.

    Args:
        query (str): The ingredients provided by the user.

    Returns:
        Dict[str, Any]: A dictionary containing the search results.
    """
    return tavily_client.search(query)

model = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0,
    max_tokens=None,
    reasoning_format="parsed",
    timeout=None,
    max_retries=2,
)

agent = create_agent(
    model=model,
    tools=[web_search],
    system_prompt=system_prompt,
    checkpointer=InMemorySaver()
)

config = {"configurable": {"thread_id": "1"}}

response = agent.invoke(
    {"messages": [HumanMessage(content="I have some leftover chicken and rice. What can I make?")]},
    config
)

# Ensure recipe responses containing Unicode characters can be printed on Windows.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
print(response)