from typing import List, Dict, Union, AsyncIterable, Optional\\\nfrom multi_agent_orchestrator.types import ConversationMessage, ParticipantRole\\\\\nfrom multi_agent_orchestrator.utils.logger import Logger\\\\\nfrom .agent import Agent, AgentOptions\\\\\n\\nclass ChainAgentOptions(AgentOptions):\\\\\n    def __init__(self, agents: List[Agent], default_output: Optional[str] = None, **kwargs):\\\\\n        super().__init__(**kwargs)\\\\\n        self.agents = agents\\\\\n        self.default_output = default_output or "No output generated from the chain."\\\\\n        if len(self.agents) == 0:\\\\\n            raise ValueError("ChainAgent requires at least one agent in the chain.")\\\n\\nclass ChainAgent(Agent):\\\\\n    def __init__(self, options: ChainAgentOptions):\\\\\n        super().__init__(options)\\\\\n        self.agents = options.agents\\\\\n        self.default_output = options.default_output\\\\\n        if len(self.agents) == 0:\\\\\n            raise ValueError("ChainAgent requires at least one agent in the chain.")\\\n\\n    async def process_request(\\\n        self, \\\\\n        input_text: str, \\\\\n        user_id: str, \\\\\n        session_id: str, \\\\\n        chat_history: List[ConversationMessage], \\\\\n        additional_params: Optional[Dict[str, str]] = None\\\\\n    ) -> Union[ConversationMessage, AsyncIterable[any]]:\\\\\n        current_input = input_text\\\\\n        final_response: Union[ConversationMessage, AsyncIterable[any]]\\\\\n\\n        for i, agent in enumerate(self.agents):\\\\\n            is_last_agent = i == len(self.agents) - 1\\\\\n            try:\\\\\n                response = await agent.process_request(\\\n                    current_input, \\\\\n                    user_id, \\\\\n                    session_id, \\\\\n                    chat_history, \\\\\n                    additional_params \\\\\n                )\\\\\n                if self.is_conversation_message(response):\\\\\n                    if response.content and 'text' in response.content[0]:\\\\\n                        current_input = response.content[0]['text']\\\\\n                        final_response = response\\\\\n                    else:\\\\\n                        Logger.logger.warning(f"Agent {agent.name} returned no text content.")\\\\\\\\n                        return self.create_default_response()\\\\\n                elif self.is_async_iterable(response):\\\\\n                    if not is_last_agent:\\\\\n                        Logger.logger.warning(f"Intermediate agent {agent.name} returned a streaming response, which is not allowed.")\\\\\\\\n                        return self.create_default_response()\\\\\n                    final_response = response\\\\\n                else:\\\\\n                    Logger.logger.warning(f"Agent {agent.name} returned an invalid response type.")\\\\\\\\n                    return self.create_default_response()\\\\\n\\n                if not is_last_agent and not self.is_conversation_message(final_response):\\\\\n                    Logger.logger.error(f"Expected non-streaming response from intermediate agent {agent.name}")\\\\\\\\n                    return self.create_default_response()\\\\\n\\n            except Exception as error:\\\\\n                Logger.logger.error(f"Error processing request with agent {agent.name}:", error)\\\\\\\\n                return self.create_default_response()\\\\\n\\n        return final_response\\\\\n\\n    @staticmethod\\\\\n    def is_async_iterable(obj: any) -> bool:\\\\\n        return hasattr(obj, '__aiter__')\\\\\\\\n\\n    @staticmethod\\\\\n    def is_conversation_message(response: any) -> bool:\\\\\n        return (\\\\\n            isinstance(response, ConversationMessage) and \\\\\n            hasattr(response, 'role') and \\\\\n            hasattr(response, 'content') and \\\\\n            isinstance(response.content, list) \\\\\n        )\\\\\\\\n\\n    def create_default_response(self) -> ConversationMessage:\\\\\n        return ConversationMessage(\\\\\n            role=ParticipantRole.ASSISTANT, \\\\\n            content=[{"text": self.default_output}]\\\\\n        )\\\\\\\\n