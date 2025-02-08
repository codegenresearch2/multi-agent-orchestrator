from typing import List, Dict, Optional, Any\\\nfrom multi_agent_orchestrator.types import ConversationMessage, ParticipantRole, BEDROCK_MODEL_ID_CLAUDE_3_HAIKU\\\\\nfrom multi_agent_orchestrator.utils import conversation_to_dict, Logger\\\\\nfrom dataclasses import dataclass\\\\\nfrom .agent import Agent, AgentOptions\\\\\nimport boto3\\\\n\\\\\n@dataclass\\\\\nclass BedrockTranslatorAgentOptions(AgentOptions):\\\\\n    source_language: Optional[str] = None\\\\\n    target_language: Optional[str] = None\\\\\n    inference_config: Optional[Dict[str, Any]] = None\\\\\n    model_id: Optional[str] = None\\\\\n    region: Optional[str] = None\\\\\n\\\\\nclass BedrockTranslatorAgent(Agent):\\\\\n    def __init__(self, options: BedrockTranslatorAgentOptions):\\\\\n        super().__init__(options)\\\\\n        self.source_language = options.source_language\\\\\n        self.target_language = options.target_language or 'English'\\\\\n        self.model_id = options.model_id or BEDROCK_MODEL_ID_CLAUDE_3_HAIKU\\\\\n        self.client = boto3.client('bedrock-runtime', region_name=options.region)\\\\\n\\\\\n        # Default inference configuration\\\\\n        self.inference_config: Dict[str, Any] = options.inference_config or {\\\\\\n            'maxTokens': 1000,\\\\\\n            'temperature': 0.0,\\\\\\n            'topP': 0.9,\\\\\\n            'stopSequences': []\\\\\\n        } \\\\\n\\\\\n        # Define the translation tool\\\\\n        self.tools = [{\\\\\\n            "toolSpec": {\\\\\\n                "name": "Translate",\\\\\\n                "description": "Translate text to target language",\\\\\\n                "inputSchema": {\\\\\\n                    "json": {\\\\\\n                        "type": "object",\\\\\\n                        "properties": {\\\\\\n                            "translation": {\\\\\\n                                "type": "string",\\\\\\n                                "description": "The translated text",\\\\\\n                            },\\\\\\\\\n                        },\\\\\\\\\n                        "required": ["translation"]\\\\\\\\\n                    }\\\\\\\\\n                }\\\\\\\\\n            }\\\\\\\\\n        }] \\\\\\\\\n\\\\\n    async def process_request(self,\\\\\\n                              input_text: str,\\\\\\n                              user_id: str,\\\\\\n                              session_id: str,\\\\\\n                              chat_history: List[ConversationMessage],\\\\\\n                              additional_params: Optional[Dict[str, str]] = None) -> ConversationMessage:\\\\\\n        # Check if input is a number and return it as-is if true\\\\\\n        if input_text.isdigit():\\\\\\n            return ConversationMessage(\\\\\\n                role=ParticipantRole.ASSISTANT,\\\\\\n                content=[{"text": input_text}]\\\\\\n            ) \\\\\\n\\\\\n        # Prepare user message\\\\\\n        user_message = ConversationMessage(\\\\\\n            role=ParticipantRole.USER,\\\\\\n            content=[{"text": f"<userinput>{input_text}</userinput>"}]\\\\\\n        ) \\\\\\n\\\\\n        # Construct system prompt\\\\\\n        system_prompt = "You are a translator. Translate the text within the <userinput> tags"\\\\\\n        if self.source_language:\\\\\\n            system_prompt += f" from {self.source_language} to {self.target_language}"\\\\\\n        else:\\\\\\n            system_prompt += f" to {self.target_language}"\\\\\\n        system_prompt += ". Only provide the translation using the Translate tool."\\\\\\n\\\\\n        # Prepare the converse command for Bedrock\\\\\\n        converse_cmd = {\\\\\\n            "modelId": self.model_id,\\\\\\n            "messages": [conversation_to_dict(user_message)],\\\\\\n            "system": [{"text": system_prompt}],\\\\\\n            "toolConfig": {\\\\\\n                "tools": self.tools,\\\\\\n                "toolChoice": {\\\\\\n                    "tool": {\\\\\\n                        "name": "Translate",\\\\\\\\\\\n                    },\\\\\\\\\n                },\\\\\\\\\n            },             \\\\\",\\\\\\\\\n            'inferenceConfig': self.inference_config\\\\\\\\\n        } \\\\\\\\\n\\\\\n        try:\\\\\\n            # Send request to Bedrock\\\\\\n            response = self.client.converse(**converse_cmd)\\\\\\n\\\\\n            if 'output' not in response:\\\\\\n                raise ValueError("No output received from Bedrock model")\\\\\\n\\\\\n            if response['output'].get('message', {}).get('content'):\\\\\\n                response_content_blocks = response['output']['message']['content']\\\\\\n\\\\\n                for content_block in response_content_blocks:\\\\\\n                    if "toolUse" in content_block:\\\\\\n                        tool_use = content_block["toolUse"]\\\\\\n                        if not tool_use:\\\\\\n                            raise ValueError("No tool use found in the response")\\\\\\n\\\\\n                        if not isinstance(tool_use.get('input'), dict) or 'translation' not in tool_use['input']:\\\\\\n                            raise ValueError("Tool input does not match expected structure")\\\\\\n\\\\\n                        translation = tool_use['input']['translation']\\\\\\n                        if not isinstance(translation, str):\\\\\\n                            raise ValueError("Translation is not a string")\\\\\\n\\\\\n                        # Return the translated text\\\\\\n                        return ConversationMessage(\\\\\\n                            role=ParticipantRole.ASSISTANT,\\\\\\n                            content=[{"text": translation}]\\\\\\n                        ) \\\\\\n\\\\\n            raise ValueError("No valid tool use found in the response")\\\\\\n        except Exception as error:\\\\\\n            Logger.error("Error processing translation request:", str(error))\\\\\\n            raise\\\\\n\\\\\n    def set_source_language(self, language: Optional[str]):\\\\\n        """Set the source language for translation"""\\\\\n        self.source_language = language\\\\\n\\\\\n    def set_target_language(self, language: str):\\\\\n        """Set the target language for translation"""\\\\\n        self.target_language = language\\\\\n