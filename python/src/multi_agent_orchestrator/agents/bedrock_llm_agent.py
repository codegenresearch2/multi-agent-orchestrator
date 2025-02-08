from typing import List, Dict, Any, AsyncIterable, Optional\\nfrom dataclasses import dataclass\\nimport re\\nimport boto3\\nfrom multi_agent_orchestrator.agents import Agent, AgentOptions\\nfrom multi_agent_orchestrator.types import (ConversationMessage, ParticipantRole, BEDROCK_MODEL_ID_CLAUDE_3_HAIKU, TemplateVariables)\\nfrom multi_agent_orchestrator.utils import conversation_to_dict, Logger\\nfrom multi_agent_orchestrator.retrievers import Retriever\\n\\n@dataclass\\nclass BedrockLLMAgentOptions(AgentOptions):\\n    streaming: Optional[bool] = None\\n    inference_config: Optional[Dict[str, Any]] = None\\n    guardrail_config: Optional[Dict[str, str]] = None\\n    retriever: Optional[Retriever] = None\\n    tool_config: Optional[Dict[str, Any]] = None\\n    custom_system_prompt: Optional[Dict[str, Any]] = None\\n\\nclass BedrockLLMAgent(Agent):\\n    def __init__(self, options: BedrockLLMAgentOptions):\\n        super().__init__(options)\\n        self.client = boto3.client('bedrock-runtime', region_name=options.region if options.region else None)\\n        self.model_id = options.model_id or BEDROCK_MODEL_ID_CLAUDE_3_HAIKU\\n        self.streaming = options.streaming\\n        self.inference_config = options.inference_config or {\"maxTokens\": 1000, \"temperature\": 0.0, \"topP\": 0.9, \"stopSequences\": []}\\n        self.guardrail_config = options.guardrail_config\\n        self.retriever = options.retriever\\n        self.tool_config = options.tool_config\\n        self.prompt_template = \"You are a {self.name}. {self.description} Provide helpful and accurate information based on your expertise. You will engage in an open-ended conversation, providing helpful and accurate information based on your expertise. The conversation will proceed as follows: - The human may ask an initial question or provide a prompt on any topic. - You will provide a relevant and informative response. - The human may then follow up with additional questions or prompts related to your previous response, allowing for a multi-turn dialogue on that topic. - Or, the human may switch to a completely new and unrelated topic at any point. - You will seamlessly shift your focus to the new topic, providing thoughtful and coherent responses based on your broad knowledge base. Throughout the conversation, you should aim to: - Understand the context and intent behind each new question or prompt. - Provide substantive and well-reasoned responses that directly address the query. - Draw insights and connections from your extensive knowledge when appropriate. - Ask for clarification if any part of the question or prompt is ambiguous. - Maintain a consistent, respectful, and engaging tone tailored to the human's communication style. - Seamlessly transition between topics as the human introduces new subjects.\"\\n        self.system_prompt = \"\"\\n        self.custom_variables = {}\\n        self.default_max_recursions = 20\\n        if options.custom_system_prompt:\\n            self.set_system_prompt(options.custom_system_prompt.get('template'), options.custom_system_prompt.get('variables'))\\n\\n    async def process_request(self, input_text: str, user_id: str, session_id: str, chat_history: List[ConversationMessage], additional_params: Optional[Dict[str, str]] = None) -> Union[ConversationMessage, AsyncIterable[Any]]:\\n        user_message = ConversationMessage(role=ParticipantRole.USER.value, content=[{'text': input_text}])\\n        conversation = [*chat_history, user_message]\n        self.update_system_prompt()\\n        system_prompt = self.system_prompt\\n        if self.retriever:\\n            response = await self.retriever.retrieve_and_combine_results(input_text)\\n            context_prompt = \"\nHere is the context to use to answer the user's question:\" + response\\n            system_prompt += context_prompt\\n        converse_cmd = {\"modelId\": self.model_id, \"messages\": conversation_to_dict(conversation), \"system\": [{{\"text\": system_prompt}}], \"inferenceConfig\": {\"maxTokens\": self.inference_config.get('maxTokens'), \"temperature\": self.inference_config.get('temperature'), \"topP\": self.inference_config.get('topP'), \"stopSequences\": self.inference_config.get('stopSequences')}}\\n        if self.guardrail_config:\\n            converse_cmd["guardrailConfig"] = self.guardrail_config\\n        if self.tool_config:\\n            converse_cmd["toolConfig"] = {'tools': self.tool_config["tool"]}\\n        if self.streaming:\\n            return await self.handle_streaming_response(converse_cmd)\\n        else:\\n            return await self.handle_single_response(converse_cmd)\\n\\n    async def handle_single_response(self, converse_input: Dict[str, Any]) -> ConversationMessage:\\n        try:\\n            response = self.client.converse(**converse_input)\\n            if 'output' not in response:\\n                raise ValueError("No output received from Bedrock model")\\n            return ConversationMessage(role=response['output']['message']['role'], content=response['output']['message']['content'])\\n        except Exception as error:\\n            Logger.error("Error invoking Bedrock model:", error)\\n            raise\\n\\n    async def handle_streaming_response(self, converse_input: Dict[str, Any]) -> ConversationMessage:\\n        try:\\n            response = self.client.converse_stream(**converse_input)\\n            llm_response = ''\\n            for chunk in response["stream"]:\\n                if "contentBlockDelta" in chunk:\\n                    content = chunk.get("contentBlockDelta", {}).get("delta", {}).get("text")\\n                    self.callbacks.on_llm_new_token(content)\\n                    llm_response = llm_response + content\\n            return ConversationMessage(role=ParticipantRole.ASSISTANT.value, content=[{'text':llm_response}])\\n        except Exception as error:\\n            Logger.error("Error getting stream from Bedrock model:", error)\\n            raise\\n\\n    def set_system_prompt(self, template: Optional[str] = None, variables: Optional[TemplateVariables] = None) -> None:\\n        if template:\\n            self.prompt_template = template\\n        if variables:\\n            self.custom_variables = variables\\n        self.update_system_prompt()\\n\\n    def update_system_prompt(self) -> None:\\n        all_variables = {**self.custom_variables}\\n        self.system_prompt = self.replace_placeholders(self.prompt_template, all_variables)\\n\\n    @staticmethod\\n    def replace_placeholders(template: str, variables: TemplateVariables) -> str:\\n        def replace(match):\\n            key = match.group(1)\\n            if key in variables:\\n                value = variables[key]\\n                return '\n'.join(value) if isinstance(value, list) else str(value)\n            return match.group(0)\\n        return re.sub(r'{{(\w+)}}', replace, template)\\n