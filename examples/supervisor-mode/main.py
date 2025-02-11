import os
import sys
import uuid
import asyncio
from dotenv import load_dotenv
from multi_agent_orchestrator.orchestrator import MultiAgentOrchestrator, OrchestratorConfig
from multi_agent_orchestrator.agents import (
    BedrockLLMAgent, BedrockLLMAgentOptions,
    AmazonBedrockAgent, AmazonBedrockAgentOptions,
    AnthropicAgent, AnthropicAgentOptions,
    LexBotAgent, LexBotAgentOptions
)
from multi_agent_orchestrator.classifiers import ClassifierResult
from multi_agent_orchestrator.types import ConversationMessage, AgentResponse
from multi_agent_orchestrator.storage import DynamoDbChatStorage
from logging import Logger, basicConfig, getLogger
from multi_agent_orchestrator.utils import get_current_date  # Assuming this is a utility function

load_dotenv()

# Initialize logging
logger: Logger = getLogger(__name__)
basicConfig(level=os.getenv('LOG_LEVEL', 'INFO'))

# Agent Initialization
tech_agent = BedrockLLMAgent(
    options=BedrockLLMAgentOptions(
        name="TechAgent",
        description="You are a tech agent. You are responsible for answering questions about tech. You are only allowed to answer questions about tech. You are not allowed to answer questions about anything else.",
        model_id="anthropic.claude-3-haiku-20240307-v1:0",
    )
)

sales_agent = BedrockLLMAgent(BedrockLLMAgentOptions(
    name="SalesAgent",
    description="You are a sales agent. You are responsible for answering questions about sales. You are only allowed to answer questions about sales. You are not allowed to answer questions about anything else.",
    model_id="anthropic.claude-3-haiku-20240307-v1:0",
))

weather_agent = BedrockLLMAgent(BedrockLLMAgentOptions(
        name="WeatherAgent",
        streaming=False,
        description="Specialized agent for giving weather forecast condition from a city.",
        tool_config={
            'tool': weather_tool_description,
            'toolMaxRecursions': 5,
            'useToolHandler': weather_tool_handler
        }
    ))
weather_agent.set_system_prompt(weather_tool_prompt)

health_agent = BedrockLLMAgent(BedrockLLMAgentOptions(
    name="HealthAgent",
    description="You are a health agent. You are responsible for answering questions about health. You are only allowed to answer questions about health. You are not allowed to answer questions about anything else.",
    model_id="anthropic.claude-3-haiku-20240307-v1:0",
))

travel_agent = BedrockLLMAgent(BedrockLLMAgentOptions(
    name="TravelAgent",
    description="You are a travel assistant agent. You are responsible for answering questions about travel, activities, sight seeing about a city and surrounding",
    model_id="anthropic.claude-3-haiku-20240307-v1:0",
))

airlines_agent = LexBotAgent(LexBotAgentOptions(name='AirlinesBot',
                                              description='Helps users book their flight. This bot works with US metric time and date.',
                                              locale_id='en_US',
                                              bot_id=os.getenv('AIRLINES_BOT_ID', None),
                                              bot_alias_id=os.getenv('AIRLINES_BOT_ALIAS_ID', None)))

claim_agent = AmazonBedrockAgent(AmazonBedrockAgentOptions(
    name="ClaimAgent",
    description="Specializes in handling claims and disputes.",
    agent_id=os.getenv('CLAIM_AGENT_ID',None),
    agent_alias_id=os.getenv('CLAIM_AGENT_ALIAS_ID',None)
))

supervisor_agent = AnthropicAgent(AnthropicAgentOptions(
    api_key=os.getenv('ANTHROPIC_API_KEY', None),
    name="SupervisorAgent",
    description="You are a supervisor agent. You are responsible for managing the flow of the conversation. You are only allowed to manage the flow of the conversation. You are not allowed to answer questions about anything else.",
    model_id="claude-3-5-sonnet-latest"
))

# Orchestrator Initialization
supervisor = MultiAgentOrchestrator(options=OrchestratorConfig(
        LOG_AGENT_CHAT=True,
        LOG_CLASSIFIER_CHAT=True,
        LOG_CLASSIFIER_RAW_OUTPUT=True,
        LOG_CLASSIFIER_OUTPUT=True,
        LOG_EXECUTION_TIMES=True,
        MAX_RETRIES=3,
        USE_DEFAULT_AGENT_IF_NONE_IDENTIFIED=True,
        MAX_MESSAGE_PAIRS_PER_AGENT=10,
    ),
    storage=DynamoDbChatStorage(
        table_name=os.getenv('DYNAMODB_CHAT_HISTORY_TABLE_NAME', None),
        region='us-east-1'
    )
)

# Add agents to the orchestrator
supervisor.add_agent(tech_agent)
supervisor.add_agent(sales_agent)
supervisor.add_agent(weather_agent)
supervisor.add_agent(health_agent)
supervisor.add_agent(travel_agent)
supervisor.add_agent(airlines_agent)
supervisor.add_agent(claim_agent)

async def handle_request(_orchestrator: MultiAgentOrchestrator, _user_input:str, _user_id:str, _session_id:str):
    classifier_result=ClassifierResult(selected_agent=supervisor_agent, confidence=1.0)

    response:AgentResponse = await _orchestrator.agent_process_request(_user_input, _user_id, _session_id, classifier_result)

    # Print metadata
    logger.info("\nMetadata:")
    logger.info(f"Selected Agent: {response.metadata.agent_name}")
    if isinstance(response, AgentResponse) and response.streaming is False:
        # Handle regular response
        if isinstance(response.output, str):
            logger.info(response.output)
        elif isinstance(response.output, ConversationMessage):
                logger.info(response.output.content[0].get('text'))

if __name__ == "__main__":

    USER_ID = str(uuid.uuid4())
    SESSION_ID = str(uuid.uuid4())

    logger.info(f"""Welcome to the interactive Multi-Agent system.\n
I'm here to assist you with your questions.
Here is the list of available agents:
- TechAgent: Anything related to technology
- SalesAgent: Weather you want to sell a boat, a car or house, I can give you advice
- HealthAgent: You can ask me about your health, diet, exercise, etc.
- AirlinesBot: I can help you book a flight
- WeatherAgent: I can tell you the weather in a given city
- TravelAgent: I can help you plan your next trip.
- ClaimAgent: Anything regarding the current claim you have or general information about them.
""")

    while True:
        # Get user input
        user_input = input("\nYou: ").strip()

        if user_input.lower() == 'quit':
            logger.info("Exiting the program. Goodbye!")
            sys.exit()

        # Run the async function
        if user_input is not None and user_input != '':
            asyncio.run(handle_request(supervisor, user_input, USER_ID, SESSION_ID))