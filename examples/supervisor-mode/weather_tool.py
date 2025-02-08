import requests\nfrom requests.exceptions import RequestException\nfrom typing import List, Dict, Any\nfrom multi_agent_orchestrator.types import ConversationMessage, ParticipantRole\nfrom multi_agent_orchestrator.utils import Tool, Tools\nimport json\n\n\n# Constants\nAPI_ENDPOINT = "https://api.open-meteo.com/v1/forecast"\n\n\n# Weather tool description\nweather_tool_description = [{\"toolSpec\": {\"name\": \"Weather_Tool\", \"description\": \"Get the current weather for a given location, based on its WGS84 coordinates.\", \"inputSchema\": {\"json\": {\"type\": \"object\", \"properties\": {\"latitude\": {\"type\": \"string\", \"description\": \"Geographical WGS84 latitude of the location.\"}, \"longitude\": {\"type\": \"string\", \"description\": \"Geographical WGS84 longitude of the location.\"}},\"required\": [\"latitude\", \"longitude\"]}}}]\n\n# Weather tool prompt\nweather_tool_prompt = \"\nYou are a weather assistant that provides current weather data for user-specified locations using only\nthe Weather_Tool, which expects latitude and longitude. Infer the coordinates from the location yourself.\nIf the user provides coordinates, infer the approximate location and refer to it in your response.\n\n- Explain your step-by-step process, and give brief updates before each step.\n- Only use the Weather_Tool for data. Never guess or make up information.\n- Repeat the tool use for subsequent requests if necessary.\n- If the tool errors, apologize, explain weather is unavailable, and suggest other options.\n- Report temperatures in °C (°F) and wind in km/h (mph). Keep weather reports concise. Sparingly use\nemojis where appropriate.\n- Only respond to weather queries. Remind off-topic users of your purpose.\n- Never claim to search online, access external data, or use tools besides Weather_Tool.\n- Complete the entire process until you have all required data before sending the complete response.\"\n\n# Function to fetch weather data\nasync def fetch_weather_data(input_data: Dict[str, str]) -> Dict[str, Any]:\n    \"\n    Fetches weather data for the given latitude and longitude using the Open-Meteo API.\n    Returns the weather data or an error message if the request fails.\n\n    :param input_data: A dictionary containing the latitude and longitude as strings.\n    :return: A dictionary containing the weather data or an error message.\n    \"\n    latitude = input_data.get(\"latitude\")\n    longitude = input_data.get(\"longitude\", \"\")\n    params = {\"latitude\": latitude, \"longitude\": longitude, \"current_weather\": True}\n\n    try:\n        response = requests.get(API_ENDPOINT, params=params)\n        weather_data = {\"weather_data\": response.json()}\n        response.raise_for_status()\n        return weather_data\n    except RequestException as e:\n        return {\"error\": str(e), \"message\": \"An error occurred while fetching weather data.\"}\n    except Exception as e:\n        return {\"error\": str(e), \"message\": \"An unexpected error occurred.\"}\n\n# Function to handle weather tool responses\nasync def weather_tool_handler(response: ConversationMessage, conversation: List[Dict[str, Any]]) -> ConversationMessage:\n    response_content_blocks = response.content\n\n    # Initialize an empty list of tool results\n    tool_results = []\n\n    if not response_content_blocks:\n        raise ValueError(\"No content blocks in response\")\n\n    for content_block in response_content_blocks:\n        if \"text\" in content_block:\n            # Handle text content if needed\n            pass\n\n        if \"toolUse\" in content_block:\n            tool_use_block = content_block[\"toolUse\"]\n            tool_use_name = tool_use_block.get(\"name\")\n\n            if tool_use_name == \"Weather_Tool\":\n                tool_response = await fetch_weather_data(tool_use_block[\"input\"])\n                tool_results.append({\"toolResult\": {\"toolUseId\": tool_use_block[\"toolUseId\"],\"content\": [tool_response]}})\n\n    # Embed the tool results in a new user message\n    message = ConversationMessage(\"role\": ParticipantRole.USER.value,\"content\": tool_results)\n\n    return message\n