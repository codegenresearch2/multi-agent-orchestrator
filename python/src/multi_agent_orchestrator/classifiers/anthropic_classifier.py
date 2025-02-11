from typing import List, Optional, Dict, Any
from anthropic import Anthropic
from multi_agent_orchestrator.utils import Logger
from multi_agent_orchestrator.utils.helpers import is_tool_input
from multi_agent_orchestrator.types import ConversationMessage
from multi_agent_orchestrator.classifiers import Classifier, ClassifierResult
import logging

logging.getLogger("httpx").setLevel(logging.WARNING)

ANTHROPIC_MODEL_ID_CLAUDE_3_5_SONNET = "claude-3-5-sonnet-20240620"

class AnthropicClassifierOptions:
    """
    A class to hold the options for the AnthropicClassifier.
    
    Attributes:
        api_key (str): The API key for the Anthropic service.
        model_id (Optional[str]): The ID of the model to use. Defaults to None.
        inference_config (Optional[Dict[str, Any]]): Configuration for the inference.
    """
    def __init__(self, api_key: str, model_id: Optional[str] = None, inference_config: Optional[Dict[str, Any]] = None):
        self.api_key = api_key
        self.model_id = model_id
        self.inference_config = inference_config if inference_config is not None else {}

class AnthropicClassifier(Classifier):
    """
    A classifier that uses the Anthropic API to process requests.
    """
    def __init__(self, options: AnthropicClassifierOptions):
        """
        Initializes the AnthropicClassifier.
        
        Args:
            options (AnthropicClassifierOptions): The options for the classifier.
        """
        super().__init__()
        if not options.api_key:
            raise ValueError("Anthropic API key is required")
        self.client = Anthropic(api_key=options.api_key)
        self.model_id = options.model_id or ANTHROPIC_MODEL_ID_CLAUDE_3_5_SONNET

        # Default inference configuration
        default_max_tokens = 1000
        self.inference_config = {
            'max_tokens': options.inference_config.get('max_tokens', default_max_tokens) or default_max_tokens,
            'temperature': options.inference_config.get('temperature', 0.0) or 0.0,
            'top_p': options.inference_config.get('top_p', 0.9) or 0.9,
            'stop_sequences': options.inference_config.get('stop_sequences', []) or [],
        }

        # Add your system prompt here
        self.system_prompt = "You are an AI assistant."

        # Add your tools here
        self.tools: List[Dict] = [
            {
                'name': 'analyzePrompt',
                'description': 'Analyze the user input and provide structured output',
                'input_schema': {
                    'type': 'object',
                    'properties': {
                        'userinput': {
                            'type': 'string',
                            'description': 'The original user input',
                        },
                        'selected_agent': {
                            'type': 'string',
                            'description': 'The name of the selected agent',
                        },
                        'confidence': {
                            'type': 'number',
                            'description': 'Confidence level between 0 and 1',
                        },
                    },
                    'required': ['userinput', 'selected_agent', 'confidence'],
                },
            }
        ]

    async def process_request(self, input_text: str, chat_history: List[ConversationMessage]) -> ClassifierResult:
        """
        Processes a request using the Anthropic API.
        
        Args:
            input_text (str): The input text to process.
            chat_history (List[ConversationMessage]): The chat history.
        
        Returns:
            ClassifierResult: The result of the classification.
        """
        user_message = {"role": "user", "content": input_text}
        try:
            response = self.client.messages.create(
                model=self.model_id,
                max_tokens=self.inference_config['max_tokens'],
                messages=[user_message],
                system=self.system_prompt,
                temperature=self.inference_config['temperature'],
                top_p=self.inference_config['top_p'],
                tools=self.tools
            )
            tool_use = next((c for c in response.content if c.type == "tool_use"), None)
            if not tool_use:
                raise ValueError("No tool use found in the response")
            if not is_tool_input(tool_use.input):
                raise ValueError("Tool input does not match expected structure")
            intent_classifier_result = ClassifierResult(
                selected_agent=self.get_agent_by_id(tool_use.input['selected_agent']),
                confidence=float(tool_use.input['confidence'])
            )
            return intent_classifier_result
        except Exception as error:
            Logger.error(f"Error processing request: {str(error)}")
            raise error