import os\nfrom typing import List, Optional, Dict, Any\nimport boto3\nfrom botocore.exceptions import BotoCoreError, ClientError\nfrom multi_agent_orchestrator.types import BEDROCK_MODEL_ID_CLAUDE_3_5_SONNET, Classifier\nfrom multi_agent_orchestrator.utils.helpers import is_tool_input\nfrom multi_agent_orchestrator.utils import Logger\nfrom multi_agent_orchestrator.types import ConversationMessage, ParticipantRole\n\nclass BedrockClassifierOptions:\n    def __init__(self, model_id: Optional[str] = None, region: Optional[str] = None, inference_config: Optional[Dict] = None):\n        self.model_id = model_id\n        self.region = region or os.environ.get('REGION')\n        self.inference_config = inference_config if inference_config is not None else {} \n\nclass BedrockClassifier(Classifier):\n    def __init__(self, options: BedrockClassifierOptions):\n        super().__init__()\n        self.region = options.region or os.environ.get('REGION')\n        self.client = boto3.client('bedrock-runtime', region_name=self.region) \n        self.model_id = options.model_id or BEDROCK_MODEL_ID_CLAUDE_3_5_SONNET \n        self.system_prompt: str\n        self.inference_config = { \n            'maxTokens': options.inference_config.get('maxTokens', 1000), \n            'temperature': options.inference_config.get('temperature', 0.0), \n            'topP': options.inference_config.get('top_p', 0.9), \n            'stopSequences': options.inference_config.get('stop_sequences', []) \n        } \n        self.tools = [ \n            { \n                "toolSpec": { \n                    "name": "analyzePrompt", \n                    "description": "Analyze the user input and provide structured output", \n                    "inputSchema": { \n                        "json": { \n                            "type": "object", \n                            "properties": { \n                                "userinput": { \n                                    "type": "string", \n                                    "description": "The original user input", \n                                }, \n                                "selected_agent": { \n                                    "type": "string", \n                                    "description": "The name of the selected agent", \n                                }, \n                                "confidence": { \n                                    "type": "number", \n                                    "description": "Confidence level between 0 and 1", \n                                }, \n                            }, \n                            "required": ["userinput", "selected_agent", "confidence"] \n                        }, \n                    }, \n                }, \n            }, \n        ] \n\n    async def process_request(self, \n                              input_text: str, \n                              chat_history: List[ConversationMessage]) -> ClassifierResult: \n        user_message = ConversationMessage( \n            role=ParticipantRole.USER.value, \n            content=[{"text": input_text}] \n        ) \n\n        converse_cmd = { \n            "modelId": self.model_id, \n            "messages": [user_message.__dict__], \n            "system": [{"text": self.system_prompt}], \n            "toolConfig": { \n                "tools": self.tools, \n                "toolChoice": { \n                    "tool": { \n                        "name": "analyzePrompt", \n                    }, \n                }, \n            }, \n            "inferenceConfig": { \n                "maxTokens": self.inference_config['maxTokens'], \n                "temperature": self.inference_config['temperature'], \n                "topP": self.inference_config['topP'], \n                "stopSequences": self.inference_config['stopSequences'], \n            }, \n        } \n\n        try: \n            response = self.client.converse(**converse_cmd) \n\n            if not response.get('output'): \n                raise ValueError("No output received from Bedrock model") \n\n            if response['output'].get('message', {}).get('content'): \n                response_content_blocks = response['output']['message']['content'] \n\n                for content_block in response_content_blocks: \n                    if 'toolUse' in content_block: \n                        tool_use = content_block['toolUse'] \n                        if not tool_use: \n                            raise ValueError("No tool use found in the response") \n\n                        if not is_tool_input(tool_use['input']): \n                            raise ValueError("Tool input does not match expected structure") \n\n                        intent_classifier_result: ClassifierResult = ClassifierResult( \n                            selected_agent=self.get_agent_by_id(tool_use['input']['selected_agent']), \n                            confidence=float(tool_use['input']['confidence']) \n                        ) \n                        return intent_classifier_result \n\n            raise ValueError("No valid tool use found in the response") \n\n        except (BotoCoreError, ClientError) as error: \n            Logger.error(f"Error processing request: {str(error)}") \n            raise