import json
from typing import Any
from tool import ToolResult
from multi_agent_orchestrator.types import ParticipantRole
from duckduckgo_search import DDGS
from multi_agent_orchestrator.utils import Logger

try:
    from multi_agent_orchestrator.agents import BedrockLLMAgent
except ImportError:
    class BedrockLLMAgent:
        pass

async def tool_handler(response: Any, conversation: list[dict[str, Any]],) -> Any:
    if not response.content:
        raise ValueError("No content blocks in response")

    tool_results = []
    content_blocks = response.content

    for block in content_blocks:
        tool_use_block = block.get("toolUse") if "toolUse" in block else None
        if not tool_use_block:
            continue

        tool_name = tool_use_block.get("name")
        tool_id = tool_use_block.get("toolUseId")
        input_data = tool_use_block.get("input")

        if tool_name == "search_web":
            result = await search_web(input_data.get('query'))
        else:
            result = f"Unknown tool use name: {tool_name}"
            Logger.error(result)

        tool_result = ToolResult(tool_id, result)
        formatted_result = tool_result.to_bedrock_format()
        tool_results.append(formatted_result)

    return tool_results

async def search_web(query: str, num_results: int = 2) -> str:
    """
    Search Web using the DuckDuckGo. Returns the search results.

    Args:
        query (str): The query to search for.
        num_results (int): The number of results to return.

    Returns:
        str: The search results from Google.
            Keys:
                - 'search_results': List of organic search results.
                - 'recipes_results': List of recipes search results.
                - 'shopping_results': List of shopping search results.
                - 'knowledge_graph': The knowledge graph.
                - 'related_questions': List of related questions.
    """
    try:
        Logger.info(f"Searching DDG for: {query}")
        search = DDGS().text(query, max_results=num_results)
        return '\n'.join(result.get('body','') for result in search)
    except Exception as e:
        Logger.error(f"Error searching for the query {query}: {e}")
        return f"Error searching for the query {query}: {e}"