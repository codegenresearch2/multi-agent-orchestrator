from typing import List, Dict, Union, Optional, Callable, Any\\\nfrom multi_agent_orchestrator.types import ConversationMessage, ParticipantRole\\\\\\\nfrom multi_agent_orchestrator.utils.logger import Logger\\\\\nfrom .agent import Agent, AgentOptions\\\\\nimport boto3\\\\\nfrom botocore.config import Config\\\\\n\\\\n# Type alias for CheckFunction\\\\\nCheckFunction = Callable[[str], str]\\\\\n\\\\nclass ComprehendFilterAgentOptions(AgentOptions):\\\\\n    def __init__(self, \\\\\n                 enable_sentiment_check: bool = True,\\\\\n                 enable_pii_check: bool = True,\\\\\n                 enable_toxicity_check: bool = True,\\\\\n                 sentiment_threshold: float = 0.7,\\\\\n                 toxicity_threshold: float = 0.7,\\\\\n                 allow_pii: bool = False,\\\\\n                 language_code: str = 'en', \\\\\n                 **kwargs):\\\\\n        super().__init__(**kwargs)\\\\\n        self.enable_sentiment_check = enable_sentiment_check\\\\\n        self.enable_pii_check = enable_pii_check\\\\\n        self.enable_toxicity_check = enable_toxicity_check\\\\\n        self.sentiment_threshold = sentiment_threshold\\\\\n        self.toxicity_threshold = toxicity_threshold\\\\\n        self.allow_pii = allow_pii\\\\\n        self.language_code = language_code\\\\\n\\\\n    def validate_language_code(self, language_code: Optional[str]) -> Optional[str]:\\\\\n        if not language_code:\\\\\n            return None\\\\\n\\\\n        valid_language_codes = ['en', 'es', 'fr', 'de', 'it', 'pt', 'ar', 'hi', 'ja', 'ko', 'zh', 'zh-TW']\\\\\n        return language_code if language_code in valid_language_codes else None\\\\\n\\\\nclass ComprehendFilterAgent(Agent):\\\\\n    def __init__(self, options: ComprehendFilterAgentOptions):\\\\\n        super().__init__(options)\\\\\n        config = Config(region_name=options.region) if options.region else None\\\\\n        self.comprehend_client = boto3.client('comprehend', config=config)\\\\\n        self.custom_checks: List[CheckFunction] = []\\\\\n        self.enable_sentiment_check = options.enable_sentiment_check\\\\\n        self.enable_pii_check = options.enable_pii_check\\\\\n        self.enable_toxicity_check = options.enable_toxicity_check\\\\\n        self.sentiment_threshold = options.sentiment_threshold\\\\\n        self.toxicity_threshold = options.toxicity_threshold\\\\\n        self.allow_pii = options.allow_pii\\\\\n        self.language_code = self.validate_language_code(options.language_code) or 'en'\\\\\n        if not any([self.enable_sentiment_check, self.enable_pii_check, self.enable_toxicity_check]):\\\\\n            self.enable_toxicity_check = True\\\\\n\\\\n    async def process_request(self,\\\\\n                              input_text: str,\\\\\n                              user_id: str,\\\\\n                              session_id: str,\\\\\n                              chat_history: List[ConversationMessage],\\\\\n                              additional_params: Optional[Dict[str, str]] = None) -> Optional[ConversationMessage]:\\\\\n        try:\\\\\n            issues: List[str] = []\\\\\n\\\\n            # Run all checks\\\\\n            sentiment_result = self.detect_sentiment(input_text) if self.enable_sentiment_check else None\\\\\n            pii_result = self.detect_pii_entities(input_text) if self.enable_pii_check else None\\\\\n            toxicity_result = self.detect_toxic_content(input_text) if self.enable_toxicity_check else None\\\\\n\\\\n            # Process results\\\\\n            if self.enable_sentiment_check and sentiment_result:\\\\\n                sentiment_issue = self.check_sentiment(sentiment_result)\\\\\n                if sentiment_issue:\\\\\n                    issues.append(sentiment_issue)\\\\\n\\\\n            if self.enable_pii_check and pii_result:\\\\\n                pii_issue = self.check_pii(pii_result)\\\\\n                if pii_issue:\\\\\n                    issues.append(pii_issue)\\\\\n\\\\n            if self.enable_toxicity_check and toxicity_result:\\\\\n                toxicity_issue = self.check_toxicity(toxicity_result)\\\\\n                if toxicity_issue:\\\\\n                    issues.append(toxicity_issue)\\\\\n\\\\n            # Run custom checks\\\\\n            for check in self.custom_checks:\\\\\n                custom_issue = await check(input_text)\\\\\n                if custom_issue:\\\\\n                    issues.append(custom_issue)\\\\\n\\\\n            if issues:\\\\\n                Logger.logger.warning(f"Content filter issues detected: {'; '.join(issues)}")\\\\\n                return None  # Return None to indicate content should not be processed further\\\\\n\\\\n            # If no issues, return the original input as a ConversationMessage\\\\\n            return ConversationMessage(\\\n                role=ParticipantRole.ASSISTANT,\\\\\n                content=[{"text": input_text}]\\\\\n            )\\\\\n\\\\n        except Exception as error:\\\\\n            Logger.logger.error("Error in ComprehendContentFilterAgent:", error)\\\\\n            raise\\\\\n\\\\n    def add_custom_check(self, check: CheckFunction):\\\\\n        self.custom_checks.append(check)\\\\\n\\\\n    def check_sentiment(self, result: Dict[str, Any]) -> Optional[str]:\\\\\n        if result['Sentiment'] == 'NEGATIVE' and result['SentimentScore']['Negative'] > self.sentiment_threshold:\\\\\n            return f"Negative sentiment detected ({result['SentimentScore']['Negative']:.2f})"\\\\\n        return None\\\\\n\\\\n    def check_pii(self, result: Dict[str, Any]) -> Optional[str]:\\\\\n        if not self.allow_pii and result.get('Entities'):\\\\\n            return f"PII detected: {', '.join(e['Type'] for e in result['Entities'])}"\\\\\n        return None\\\\\n\\\\n    def check_toxicity(self, result: Dict[str, Any]) -> Optional[str]:\\\\\n        toxic_labels = self.get_toxic_labels(result)\\\\\n        if toxic_labels:\\\\\n            return f"Toxic content detected: {', '.join(toxic_labels)}"\\\\\n        return None\\\\\n\\\\n    def detect_sentiment(self, text: str) -> Dict[str, Any]:\\\\\n        return self.comprehend_client.detect_sentiment(\\\n            Text=text,\\\\\n            LanguageCode=self.language_code\\\\\n        )\\\\\n\\\\n    def detect_pii_entities(self, text: str) -> Dict[str, Any]:\\\\\n        return self.comprehend_client.detect_pii_entities(\\\n            Text=text,\\\\\n            LanguageCode=self.language_code\\\\\n        )\\\\\n\\\\n    def detect_toxic_content(self, text: str) -> Dict[str, Any]:\\\\\n        return self.comprehend_client.detect_toxic_content(\\\n            TextSegments=[{"Text": text}], \\\\\n            LanguageCode=self.language_code\\\\\n        )\\\\\n\\\\n    def get_toxic_labels(self, toxicity_result: Dict[str, Any]) -> List[str]:\\\\\n        toxic_labels = []\\\\\n        for result in toxicity_result.get('ResultList', []):\\\\\n            for label in result.get('Labels', []):\\\\\n                if label['Score'] > self.toxicity_threshold:\\\\\n                    toxic_labels.append(label['Name'])\\\\\n        return toxic_labels\\\\\n\\\\n    def set_language_code(self, language_code: str):\\\\\n        validated_language_code = self.validate_language_code(language_code)\\\\\n        if validated_language_code:\\\\\n            self.language_code = validated_language_code\\\\\n        else:\\\\\n            raise ValueError(f"Invalid language code: {language_code}")\\\\\n