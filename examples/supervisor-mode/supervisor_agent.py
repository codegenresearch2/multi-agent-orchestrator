from typing import Optional, Any, AsyncIterable, Union
from dataclasses import dataclass, field
from enum import Enum
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
from multi_agent_orchestrator.agents import Agent, AgentOptions, BedrockLLMAgent, AnthropicAgent
from multi_agent_orchestrator.types import ConversationMessage, ParticipantRole
from multi_agent_orchestrator.utils import Logger
from multi_agent_orchestrator.storage import ChatStorage, InMemoryChatStorage
from tool import Tool, ToolResult
from datetime import datetime, timezone

class SupervisorType(Enum):
    BEDROCK = "BEDROCK"
    ANTHROPIC = "ANTHROPIC"

@dataclass
class SupervisorAgentOptions(AgentOptions):
    supervisor: Agent = None
    team: list[Agent] = field(default_factory=list)
    storage: Optional[ChatStorage] = None
    trace: Optional[bool] = None
    extra_tools: Union[list[Tool], Tool] = field(default_factory=list)

    # Hide inherited fields
    name: str = field(init=False)
    description: str = field(init=False)

class SupervisorAgent(Agent):
    """
    SupervisorAgent class.

    This class represents a supervisor agent that interacts with other agents in an environment. It inherits from the Agent class.

    Attributes:
        supervisor_tools (list[Tool]): List of tools available to the supervisor agent.
        team (list[Agent]): List of agents in the environment.
        supervisor_type (SupervisorType): Type of supervisor agent (BEDROCK or ANTHROPIC).
        user_id (str): User ID.
        session_id (str): Session ID.
        storage (ChatStorage): Chat storage for storing conversation history.
        trace (bool): Flag indicating whether to enable tracing.

    Methods:
        __init__(self, options: SupervisorAgentOptions): Initializes a SupervisorAgent instance.
        send_message(self, agent: Agent, content: str, user_id: str, session_id: str, additionalParameters: dict) -> str: Sends a message to an agent.
        send_messages(self, messages: list[dict[str, str]]) -> str: Sends messages to multiple agents in parallel.
        get_current_date(self) -> str: Gets the current date.
        supervisor_tool_handler(self, response: Any, conversation: list[dict[str, Any]]) -> Any: Handles the response from a tool.
        _process_tool(self, tool_name: str, input_data: dict) -> Any: Processes a tool based on its name.
        process_request(self, input_text: str, user_id: str, session_id: str, chat_history: list[ConversationMessage], additional_params: Optional[dict[str, str]] = None) -> Union[ConversationMessage, AsyncIterable[Any]]: Processes a user request.
    """

    def __init__(self, options: SupervisorAgentOptions):
        options.name = options.supervisor.name
        options.description = options.supervisor.description
        super().__init__(options)
        self.supervisor: Union[BedrockLLMAgent, AnthropicAgent] = options.supervisor

        self.team = options.team
        self.supervisor_type = SupervisorType(options.supervisor.type)
        if not self.supervisor.tool_config:
            all_tools = options.extra_tools if isinstance(options.extra_tools, list) else [options.extra_tools] + SupervisorAgent.supervisor_tools
            self.supervisor.tool_config = {
                'tool': [tool.to_bedrock_format() if self.supervisor_type == SupervisorType.BEDROCK else tool.to_claude_format() for tool in all_tools],
                'toolMaxRecursions': 40,
                'useToolHandler': self.supervisor_tool_handler
            }
        else:
            raise RuntimeError('Supervisor tool config already set. Please do not set it manually.')

        self.user_id = ''
        self.session_id = ''
        self.storage = options.storage or InMemoryChatStorage()
        self.trace = options.trace

    def send_message(self, agent: Agent, content: str, user_id: str, session_id: str, additionalParameters: dict) -> 'str':
        Logger.info(f"\n===>>>>> Supervisor sending  {agent.name}: {content}") if self.trace else None
        agent_chat_history = asyncio.run(self.storage.fetch_chat(user_id, session_id, agent.id)) if agent.save_chat else []
        response = asyncio.run(agent.process_request(content, user_id, session_id, agent_chat_history, additionalParameters))
        asyncio.run(self.storage.save_chat_message(user_id, session_id, agent.id, ConversationMessage(role=ParticipantRole.USER.value, content=[{'text': content}]))) if agent.save_chat else None
        asyncio.run(self.storage.save_chat_message(user_id, session_id, agent.id, ConversationMessage(role=ParticipantRole.ASSISTANT.value, content=[{'text': f"{response.content[0].get('text', '')}"}]))) if agent.save_chat else None
        Logger.info(f"\n<<<<<===Supervisor received this response from {agent.name}:\n{response.content[0].get('text', '')[:500]}...") if self.trace else None
        return f"{agent.name}: {response.content[0].get('text')}"

    async def send_messages(self, messages: list[dict[str, str]]):
        """Process all messages for all agents in parallel."""
        tasks = []

        # Create tasks for each matching agent/message pair
        for agent in self.team:
            for message in messages:
                if agent.name == message.get('recipient'):
                    # Wrap the entire send_message call in to_thread
                    task = asyncio.create_task(
                        asyncio.to_thread(
                            self.send_message,
                            agent,
                            message.get('content'),
                            self.user_id,
                            self.session_id,
                            {}
                        )
                    )
                    tasks.append(task)

        # Gather and wait for all tasks to complete
        if tasks:
            responses = await asyncio.gather(*tasks)
            return ''.join(responses)
        return ''

    async def get_current_date(self):
        Logger.info('Using Tool : get_current_date')
        return datetime.now(timezone.utc).strftime('%m/%d/%Y')  # from datetime import datetime, timezone

    async def supervisor_tool_handler(self, response: Any, conversation: list[dict[str, Any]],) -> Any:
        if not response.content:
            raise ValueError("No content blocks in response")

        tool_results = []
        content_blocks = response.content

        for block in content_blocks:
            # Determine if it's a tool use block based on platform
            tool_use_block = self._get_tool_use_block(block)
            if not tool_use_block:
                continue

            tool_name = tool_use_block.get("name")
            tool_id = tool_use_block.get("toolUseId")
            input_data = tool_use_block.get("input", {})

            # Process the tool use
            result = await self._process_tool(tool_name, input_data)

            # Create tool result
            tool_result = ToolResult(tool_id, result)

            # Format according to platform
            formatted_result = tool_result.to_bedrock_format() if self.supervisor_type == SupervisorType.BEDROCK else tool_result.to_anthropic_format()

            tool_results.append(formatted_result)

        # Create and return appropriate message format
        return ConversationMessage(
            role=ParticipantRole.USER.value,
            content=tool_results
        )

    async def _process_tool(self, tool_name: str, input_data: dict) -> Any:
        """Process tool use based on tool name."""
        if tool_name == "send_messages":
            return await self.send_messages(input_data.get('messages'))
        elif tool_name == "get_current_date":
            return await self.get_current_date()
        else:
            error_msg = f"Unknown tool use name: {tool_name}"
            Logger.error(error_msg)
            raise ValueError(error_msg)

    async def process_request(
        self,
        input_text: str,
        user_id: str,
        session_id: str,
        chat_history: list[ConversationMessage],
        additional_params: Optional[dict[str, str]] = None
    ) -> Union[ConversationMessage, AsyncIterable[Any]]:

        self.user_id = user_id
        self.session_id = session_id

        # fetch history from all agents (including supervisor)
        agents_history = await self.storage.fetch_all_chats(user_id, session_id)
        agents_memory = ''.join(
            f"{user_msg.role}:{user_msg.content[0].get('text','')}\n"
            f"{asst_msg.role}:{asst_msg.content[0].get('text','')}\n"
            for user_msg, asst_msg in zip(agents_history[::2], agents_history[1::2])
            if self.id not in asst_msg.content[0].get('text', '') # removing supervisor history from agents_memory (already part of chat_history)
        )

        # update prompt with agents memory
        self.supervisor.set_system_prompt(self.prompt_template.replace('{AGENTS_MEMORY}', agents_memory))
        # call the supervisor
        try:
            response = await self.supervisor.process_request(input_text, user_id, session_id, chat_history, additional_params)
        except Exception as e:
            Logger.error(f"Error processing request: {e}")
            return "An error occurred while processing your request."
        return response

    def _get_tool_use_block(self, block: dict) -> Union[dict, None]:
        """Extract tool use block based on platform format."""
        if self.supervisor_type == SupervisorType.BEDROCK.value and "toolUse" in block:
            return block["toolUse"]
        elif self.supervisor_type == SupervisorType.ANTHROPIC.value and block.type == "tool_use":
            return block
        return None