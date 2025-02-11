import json
from typing import Any
from tool import ToolResult
from multi_agent_orchestrator.types import ParticipantRole, ConversationMessage
from tool import Tool
from duckduckgo_search import DDGS
from multi_agent_orchestrator.utils import Logger

search_web_tool = Tool(name='search_web',
                          description='Search Web for information',
                          properties={
                              'query': {
                                  'type': 'string',
                                  'description': 'The search query'
                              }
                          },
                          required=['query'])

async def tool_handler(response: Any, conversation: list[dict[str, Any]],) -> Any:
    if not response.content:
        raise ValueError("No content blocks in response")

    tool_results = []
    content_blocks = response.content

    for block in content_blocks:
        # Streamlined approach to directly assign the `tool_use_block` variable
        tool_use_block = block.get('toolUse', {})
        tool_name = tool_use_block.get('name')
        tool_id = tool_use_block.get('toolUseId')
        input_data = tool_use_block.get('input', {})

        # Process the tool use
        if tool_name == "search_web":
            result = search_web(input_data.get('query'))
        else:
            result = f"Unknown tool use name: {tool_name}"

        # Create tool result
        tool_result = ToolResult(tool_id, result)

        # Format according to platform
        formatted_result = tool_result.to_bedrock_format()

        tool_results.append(formatted_result)

    # Create and return appropriate message format
    return ConversationMessage(
        role=ParticipantRole.USER.value,
        content=tool_results
    )

def search_web(query: str, num_results: int = 2) -> str:
    """
    Search Web using the DuckDuckGo. Returns the search results.

    Args:
        query(str): The query to search for.
        num_results(int): The number of results to return.

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
        return '\n'.join(result.get('body', '') for result in search)
    except Exception as e:
        Logger.error(f"Error searching for the query {query}: {e}")
        return f"Error searching for the query {query}: {e}"