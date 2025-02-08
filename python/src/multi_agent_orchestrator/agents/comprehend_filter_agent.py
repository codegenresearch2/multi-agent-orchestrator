from typing import List, Dict, Union, Optional, Callable, Any
from multi_agent_orchestrator.types import ConversationMessage, ParticipantRole
from multi_agent_orchestrator.utils.logger import Logger
from .agent import Agent, AgentOptions
import boto3
from botocore.config import Config

# Type alias for CheckFunction
CheckFunction = Callable[[str], str]

class ComprehendFilterAgentOptions(AgentOptions):
    def __init__(self,
                 enable_sentiment_check: bool = True,
                 enable_pii_check: bool = True,
                 enable_toxicity_check: bool = True,
                 sentiment_threshold: float = 0.7,
                 toxicity_threshold: float = 0.7,
                 allow_pii: bool = False,
                 language_code: str = 'en',
                 **kwargs):
        super().__init__(**kwargs)
        self.enable_sentiment_check = enable_sentiment_check
        self.enable_pii_check = enable_pii_check
        self.enable_toxicity_check = enable_toxicity_check
        self.sentiment_threshold = sentiment_threshold
        self.toxicity_threshold = toxicity_threshold
        self.allow_pii = allow_pii
        self.language_code = self.validate_language_code(language_code) or 'en'
        # Ensure at least one check is enabled
        if not any([self.enable_sentiment_check, self.enable_pii_check, self.enable_toxicity_check]):
            self.enable_toxicity_check = True

class ComprehendFilterAgent(Agent):
    def __init__(self, options: ComprehendFilterAgentOptions):
        super().__init__(options)
        config = Config(region_name=options.region) if options.region else None
        self.comprehend_client = boto3.client('comprehend', config=config)
        self.custom_checks = []
        self.enable_sentiment_check = options.enable_sentiment_check
        self.enable_pii_check = options.enable_pii_check
        self.enable_toxicity_check = options.enable_toxicity_check
        self.sentiment_threshold = options.sentiment_threshold
        self.toxicity_threshold = options.toxicity_threshold
        self.allow_pii = options.allow_pii
        self.language_code = self.validate_language_code(options.language_code) or 'en'

    async def process_request(self,
                              input_text: str,
                              user_id: str,
                              session_id: str,
                              chat_history: List[ConversationMessage],
                              additional_params: Optional[Dict[str, str]] = None) -> Optional[ConversationMessage]:
        try:
            issues: List[str] = []
            sentiment_result = self.detect_sentiment(input_text) if self.enable_sentiment_check else None
            pii_result = self.detect_pii_entities(input_text) if self.enable_pii_check else None
            toxicity_result = self.detect_toxic_content(input_text) if self.enable_toxicity_check else None
            if self.enable_sentiment_check and sentiment_result:
                sentiment_issue = self.check_sentiment(sentiment_result)
                if sentiment_issue:
                    issues.append(sentiment_issue)
            if self.enable_pii_check and pii_result:
                pii_issue = self.check_pii(pii_result)
                if pii_issue:
                    issues.append(pii_issue)
            if self.enable_toxicity_check and toxicity_result:
                toxicity_issue = self.check_toxicity(toxicity_result)
                if toxicity_issue:
                    issues.append(toxicity_issue)
            for check in self.custom_checks:
                custom_issue = await check(input_text)
                if custom_issue:
                    issues.append(custom_issue)
            if issues:
                Logger.logger.warning(f'Content filter issues detected: {'; '.join(issues)}')
                return None
            return ConversationMessage(
                role=ParticipantRole.ASSISTANT,
                content=[{'text': input_text}])
        except Exception as error:
            Logger.logger.error('Error in ComprehendContentFilterAgent:', error)
            raise