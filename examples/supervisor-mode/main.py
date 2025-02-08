import os\"nimport asyncio\"nimport uuid\"nimport sys\"nfrom dotenv import load_dotenv\"nimport logging\"n\"nload_dotenv()\"n\"n# Initialize logging\nlogging.basicConfig(level=logging.INFO)\"n\"n# Function to create agents\ndef create_agent(name, description, model_id):\n    return BedrockLLMAgent(BedrockLLMAgentOptions(\n        name=name,\n        description=description,\n        model_id=model_id,\n    ))\"n\"n# Create agents\ntech_agent = create_agent('TechAgent', 'You are a tech agent. You are responsible for answering questions about tech. You are only allowed to answer questions about tech. You are not allowed to answer questions about anything else.', 'anthropic.claude-3-haiku-20240307-v1:0')\nsales_agent = create_agent('SalesAgent', 'You are a sales agent. You are responsible for answering questions about sales. You are only allowed to answer questions about sales. You are not allowed to answer questions about anything else.', 'anthropic.claude-3-haiku-20240307-v1:0')\nclaim_agent = create_agent('ClaimAgent', 'Specializes in handling claims and disputes.', 'anthropic.claude-3-haiku-20240307-v1:0')\nweather_agent = create_agent('WeatherAgent', 'Specialized agent for giving weather forecast condition from a city.', 'anthropic.claude-3-haiku-20240307-v1:0')\nweather_agent.set_system_prompt(weather_tool_prompt)\nhealth_agent = create_agent('HealthAgent', 'You are a health agent. You are responsible for answering questions about health. You are only allowed to answer questions about health. You are not allowed to answer questions about anything else.', 'anthropic.claude-3-haiku-20240307-v1:0')\ntravel_agent = create_agent('TravelAgent', 'You are a travel assistant agent. You are responsible for answering questions about travel, activities, sight seesing about a city and surrounding', 'anthropic.claude-3-haiku-20240307-v1:0')\nairlines_agent = create_agent('AirlinesBot', 'Helps users book their flight. This bot works with US metric time and date.', 'anthropic.claude-3-haiku-20240307-v1:0')\n\nclass LLMAgentCallbacks(AgentCallbacks):\n    def on_llm_new_token(self, token: str) -> None:\n        print(token, end='', flush=True)\n\nasync def handle_request(_orchestrator: MultiAgentOrchestrator, _user_input: str, _user_id: str, _session_id: str):\n    response: AgentResponse = await _orchestrator.agent_process_request(_user_input, _user_id, _session_id)\n\n    print('\\nMetadata:')\n    print(f'Selected Agent: {response.metadata.agent_name}')\n    if isinstance(response, AgentResponse) and response.streaming is False:\n        if isinstance(response.output, str):\n            print(response.output)\n        elif isinstance(response.output, ConversationMessage):\n            print(response.output.content[0].get('text'))\n\nif __name__ == '__main__':\n    orchestrator = MultiAgentOrchestrator(options=OrchestratorConfig(\n        LOG_AGENT_CHAT=True,\n        LOG_CLASSIFIER_CHAT=True,\n        LOG_CLASSIFIER_RAW_OUTPUT=True,\n        LOG_CLASSIFIER_OUTPUT=True,\n        LOG_EXECUTION_TIMES=True,\n        MAX_RETRIES=3,\n        USE_DEFAULT_AGENT_IF_NONE_IDENTIFIED=True,\n        MAX_MESSAGE_PAIRS_PER_AGENT=10,\n    ),\n        storage=DynamoDbChatStorage(\n            table_name=os.getenv('DYNAMODB_CHAT_HISTORY_TABLE_NAME', None),\n            region='us-east-1'\n        )\n    )\n\n    USER_ID = str(uuid.uuid4())\n    SESSION_ID = str(uuid.uuid4())\n\n    print('Welcome to the interactive Multi-Agent system.')\n    print('I\'m here to assist you with your questions.')\n    print('Here is the list of available agents:')\n    print('- TechAgent: Anything related to technology')\n    print('- SalesAgent: Weather you want to sell a boat, a car or house, I can give you advice')\n    print('- HealthAgent: You can ask me about your health, diet, exercise, etc.')\n    print('- AirlinesBot: I can help you book a flight')\n    print('- WeatherAgent: I can tell you the weather in a given city')\n    print('- TravelAgent: I can help you plan your next trip.')\n    print('- ClaimAgent: Anything regarding the current claim you have or general information about them.')\n\n    while True:\n        user_input = input('\\nYou: ').strip() \n        if user_input.lower() == 'quit':\n            print('Exiting the program. Goodbye!') \n            sys.exit() \n        if user_input is not None and user_input != '':\n            asyncio.run(handle_request(orchestrator, user_input, USER_ID, SESSION_ID))\n