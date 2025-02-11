import os
from typing import List, Optional, Dict, Any
import boto3
from botocore.exceptions import BotoCoreError, ClientError
from multi_agent_orchestrator.utils.helpers import is_tool_input
from multi_agent_orchestrator.utils import Logger
from multi_agent_orchestrator.types import ConversationMessage, ParticipantRole, BEDROCK_MODEL_ID_CLAUDE_3_5_SONNET
from multi_agent_orchestrator.classifiers import Classifier, ClassifierResult


class BedrockClassifierOptions:
    def __init__(
        self,
        model_id: Optional[str] = None,
        region: Optional[str] = None,
        inference_config: Optional[Dict[str, Any]] = None
    ):
        self.model_id = model_id
        self.region = region or os.environ.get('REGION', 'us-east-1')
        self.inference_config = inference_config if inference_config is not None else {}


class BedrockClassifier(Classifier):
    def __init__(self, options: BedrockClassifierOptions):
        super().__init__()
        self.region = options.region
        self.client = boto3.client('bedrock-runtime', region_name=self.region)
        self.model_id = options.model_id or BEDROCK_MODEL_ID_CLAUDE_3_5_SONNET
        self.system_prompt = "You are an AI assistant."  # Initialized here
        self.inference_config = {
            'max_tokens': options.inference_config.get('max_tokens', 1000),
            'temperature': options.inference_config.get('temperature', 0.0),
            'top_p': options.inference_config.get('top_p', 0.9),
            'stop_sequences': options.inference_config.get('stop_sequences', [])
        }
        self.tools = [
            {
                "toolSpec": {
                    "name": "analyzePrompt",
                    "description": "Analyze the user input and provide structured output",
                    "inputSchema": {
                        "json": {
                            "type": "object",
                            "properties": {
                                "userinput": {
                                    "type": "string",
                                    "description": "The original user input",
                                },
                                "selected_agent": {
                                    "type": "string",
                                    "description": "The name of the selected agent",
                                },
                                "confidence": {
                                    "type": "number",
                                    "description": "Confidence level between 0 and 1",
                                },
                            },
                            "required": ["userinput", "selected_agent", "confidence"],
                        },
                    },
                },
            },
        ]


    async def process_request(self,
                              input_text: str,
                              chat_history: List[ConversationMessage]) -> ClassifierResult:
        user_message = ConversationMessage(
            role=ParticipantRole.USER.value,  # Using the correct attribute
            content=[{"text": input_text}]
        )

        converse_cmd = {
            "modelId": self.model_id,
            "messages": [user_message.__dict__],
            "system": self.system_prompt,
            "tools": self.tools,
            "toolChoice": {
                "tool": {
                    "name": "analyzePrompt",
                },
            },
            "max_tokens": self.inference_config['max_tokens'],
            "temperature": self.inference_config['temperature'],
            "top_p": self.inference_config['top_p'],
            "stop_sequences": self.inference_config['stop_sequences'],
        }

        try:
            response = self.client.converse(**converse_cmd)

            if not response.get('output'):
                raise ValueError("No output received from Bedrock model")

            if response['output'].get('message', {}).get('content'):
                response_content_blocks = response['output']['message']['content']

                for content_block in response_content_blocks:
                    if 'toolUse' in content_block:
                        tool_use = content_block['toolUse']
                        if not tool_use:
                            raise ValueError("No tool use found in the response")

                        if not is_tool_input(tool_use['input']):
                            raise ValueError("Tool input does not match expected structure")

                        intent_classifier_result: ClassifierResult = ClassifierResult(
                            selected_agent=self.get_agent_by_id(tool_use['input']['selected_agent']),
                            confidence=float(tool_use['input']['confidence'])
                        )
                        return intent_classifier_result

            raise ValueError("No valid tool use found in the response")

        except (BotoCoreError, ClientError) as error:
            Logger.error(f"Error processing request: {str(error)}")
            raise