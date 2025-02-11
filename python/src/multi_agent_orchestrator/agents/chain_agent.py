from typing import List, Dict, Union, AsyncIterable, Optional
from multi_agent_orchestrator.types import ConversationMessage, ParticipantRole
from multi_agent_orchestrator.utils.logger import Logger
from .agent import Agent, AgentOptions

class ChainAgentOptions(AgentOptions):
    def __init__(self, agents: List[Agent], default_output: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)
        self.agents = agents
        self.default_output = default_output or "No output generated from the chain."
        if not self.agents:
            raise ValueError("ChainAgent requires at least one agent in the chain.")

class ChainAgent(Agent):
    def __init__(self, options: ChainAgentOptions):
        super().__init__(options)
        self.agents = options.agents
        self.default_output = options.default_output

    async def process_request(
        self,
        input_text: str,
        user_id: str,
        session_id: str,
        chat_history: List[ConversationMessage],
        additional_params: Optional[Dict[str, str]] = None
    ) -> Union[ConversationMessage, AsyncIterable[any]]:
        current_input = input_text
        final_response: Union[ConversationMessage, AsyncIterable[any]]

        for i, agent in enumerate(self.agents):
            is_last_agent = i == len(self.agents) - 1
            try:
                response = await agent.process_request(
                    current_input,
                    user_id,
                    session_id,
                    chat_history,
                    additional_params
                )
                if self.is_conversation_message(response):
                    if response.content and 'text' in response.content[0]:
                        current_input = response.content[0]['text']
                        final_response = response
                    else:
                        Logger.log_error(f"Agent {agent.name} returned no text content.")
                        return self.create_default_response()
                elif self.is_async_iterable(response):
                    if not is_last_agent:
                        Logger.log_error(f"Intermediate agent {agent.name} returned a streaming response, which is not allowed.")
                        return self.create_default_response()
                    final_response = response
                else:
                    Logger.log_error(f"Agent {agent.name} returned an invalid response type.")
                    return self.create_default_response()

                if not is_last_agent and not self.is_conversation_message(final_response):
                    Logger.log_error(f"Expected non-streaming response from intermediate agent {agent.name}")
                    return self.create_default_response()

            except Exception as error:
                Logger.log_error(f"Error processing request with agent {agent.name}: {error}")
                return self.create_default_response()

        return final_response

    @staticmethod
    def is_async_iterable(obj: any) -> bool:
        return hasattr(obj, '__aiter__')

    @staticmethod
    def is_conversation_message(response: any) -> bool:
        return (
            isinstance(response, ConversationMessage) and
            hasattr(response, 'role') and
            hasattr(response, 'content') and
            isinstance(response.content, list)
        )

    def create_default_response(self) -> ConversationMessage:
        return ConversationMessage(
            role=ParticipantRole.ASSISTANT,
            content=[{"text": self.default_output}]
        )