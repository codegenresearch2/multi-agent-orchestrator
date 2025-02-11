from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
import boto3
from multi_agent_orchestrator.agents import Agent, AgentOptions
from multi_agent_orchestrator.types import ConversationMessage, ParticipantRole
from multi_agent_orchestrator.utils import Logger


@dataclass
class BedrockLLMAgentOptions(AgentOptions):
    streaming: Optional[bool] = None
    inference_config: Optional[Dict[str, Any]] = None
    guardrail_config: Optional[Dict[str, str]] = None
    tool_config: Optional[Dict[str, Any]] = None
    custom_system_prompt: Optional[Dict[str, Any]] = None


class BedrockLLMAgent(Agent):
    def __init__(self, options: BedrockLLMAgentOptions):
        super().__init__(options)
        self.client = boto3.client('bedrock-runtime', region_name=options.region)
        self.model_id = options.model_id
        self.streaming = options.streaming
        self.inference_config = options.inference_config or {
            'maxTokens': 1000,
            'temperature': 0.0,
            'topP': 0.9,
            'stopSequences': []
        }
        self.guardrail_config = options.guardrail_config
        self.tool_config = options.tool_config
        self.prompt_template = f"""You are a {self.name}.
        {self.description}
        Provide helpful and accurate information based on your expertise.
        You will engage in an open-ended conversation,
        providing helpful and accurate information based on your expertise.
        The conversation will proceed as follows:
        - The human may ask an initial question or provide a prompt on any topic.
        - You will provide a relevant and informative response.
        - The human may then follow up with additional questions or prompts related to your previous
        response, allowing for a multi-turn dialogue on that topic.
        - Or, the human may switch to a completely new and unrelated topic at any point.
        - You will seamlessly shift your focus to the new topic, providing thoughtful and
        coherent responses based on your broad knowledge base.
        Throughout the conversation, you should aim to:
        - Understand the context and intent behind each new question or prompt.
        - Provide substantive and well-reasoned responses that directly address the query.
        - Draw insights and connections from your extensive knowledge when appropriate.
        - Ask for clarification if any part of the question or prompt is ambiguous.
        - Maintain a consistent, respectful, and engaging tone tailored
        to the human's communication style.
        - Seamlessly transition between topics as the human introduces new subjects."""
        self.system_prompt = ""
        self.custom_variables = {}
        self.default_max_recursions = 20

        if options.custom_system_prompt:
            self.set_system_prompt(
                options.custom_system_prompt.get('template'),
                options.custom_system_prompt.get('variables')
            )

    async def process_request(
        self,
        input_text: str,
        user_id: str,
        session_id: str,
        chat_history: List[ConversationMessage],
        additional_params: Optional[Dict[str, str]] = None
    ) -> Union[ConversationMessage, Any]:
        user_message = ConversationMessage(
            role=ParticipantRole.USER.value,
            content=[{'text': input_text}]
        )
        conversation = [*chat_history, user_message]
        self.update_system_prompt()
        system_prompt = self.system_prompt

        if self.tool_config:
            continue_with_tools = True
            final_message = ConversationMessage(role=ParticipantRole.USER.value, content=[])
            max_recursions = self.tool_config.get('toolMaxRecursions', self.default_max_recursions)

            while continue_with_tools and max_recursions > 0:
                bedrock_response = await self.handle_single_response(conversation)
                conversation.append(bedrock_response)

                if any('toolUse' in content for content in bedrock_response.content):
                    await self.tool_config['useToolHandler'](bedrock_response, conversation)
                else:
                    continue_with_tools = False
                    final_message = bedrock_response

                max_recursions -= 1

            return final_message

        if self.streaming:
            return await self.handle_streaming_response(conversation)

        return await self.handle_single_response(conversation)

    async def handle_single_response(self, conversation: List[ConversationMessage]) -> ConversationMessage:
        try:
            response = self.client.converse(**{
                'modelId': self.model_id,
                'messages': [msg.to_dict() for msg in conversation],
                'system': [{'text': self.system_prompt}],
                'inferenceConfig': self.inference_config
            })
            if 'output' not in response:
                raise ValueError("No output received from Bedrock model")
            return ConversationMessage(
                role=response['output']['message']['role'],
                content=response['output']['message']['content']
            )
        except Exception as error:
            Logger.error(f"Error invoking Bedrock model: {str(error)}")
            raise

    async def handle_streaming_response(self, conversation: List[ConversationMessage]) -> ConversationMessage:
        try:
            response = self.client.converse_stream(**{
                'modelId': self.model_id,
                'messages': [msg.to_dict() for msg in conversation],
                'system': [{'text': self.system_prompt}],
                'inferenceConfig': self.inference_config
            })
            llm_response = ''
            for chunk in response["stream"]:
                if "contentBlockDelta" in chunk:
                    content = chunk.get("contentBlockDelta", {}).get("delta", {}).get("text")
                    llm_response += content
            return ConversationMessage(role=ParticipantRole.ASSISTANT.value, content=[{'text': llm_response}])
        except Exception as error:
            Logger.error(f"Error getting stream from Bedrock model: {str(error)}")
            raise

    def set_system_prompt(self, template: Optional[str] = None, variables: Optional[Dict[str, str]] = None) -> None:
        if template:
            self.prompt_template = template
        if variables:
            self.custom_variables = variables
        self.update_system_prompt()

    def update_system_prompt(self) -> None:
        all_variables = {**self.custom_variables}
        self.system_prompt = self.replace_placeholders(self.prompt_template, all_variables)

    @staticmethod
    def replace_placeholders(template: str, variables: Dict[str, str]) -> str:
        def replace(match):
            key = match.group(1)
            if key in variables:
                value = variables[key]
                return '\n'.join(value) if isinstance(value, list) else str(value)
            return match.group(0)

        return re.sub(r'{{(\w+)}}', replace, template)