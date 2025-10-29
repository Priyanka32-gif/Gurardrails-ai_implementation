
from langchain_core.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults

# Create simple tools
@tool
def search_web(query: str) -> str:
    """Search the web for information."""
    # Using Tavily for web search
    search = TavilySearchResults(max_results=3)
    results = search.invoke(query)
    return str(results)


# @tool
# def write_summary(content: str) -> str:
#     """Write a summary of the provided content."""
#     # Simple summary generation
#     summary = f"Summary of findings:\n\n{content[:500]}..." 
#     return summary

