import logging
from dataclasses import dataclass, fields
from typing import List, Dict, Any, Optional
import asyncio

# Configure logging
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

@dataclass
class Configuration:
    param1: str
    param2: int

    def validate(self):
        if not self.param1:
            raise ValueError('param1 must be provided')
        if self.param2 <= 0:
            raise ValueError('param2 must be a positive integer')

class MyClass:
    def __init__(self, config: Configuration):
        self.config = config.validate() if config else Configuration(param1='default', param2=1)

    async def process_data_async(self, data: List[int]) -> List[int]:
        '''Asynchronously processes the input data and returns the processed data.'''        
        try:
            processed_data = [item * 2 for item in data]  # Example processing
            await asyncio.sleep(1)  # Simulate async operation
            return processed_data
        except Exception as e:
            logger.error(f'Error processing data: {str(e)}')
            raise

    def process_data(self, data: List[int]) -> List[int]:
        '''Processes the input data and returns the processed data.'''        
        try:
            processed_data = [item * 2 for item in data]  # Example processing
            return processed_data
        except Exception as e:
            logger.error(f'Error processing data: {str(e)}')
            raise

# Example usage
if __name__ == '__main__':
    config = Configuration(param1='example', param2=123)
    instance = MyClass(config)
    data = [1, 2, 3, 4]
    try:
        result = instance.process_data(data)  # Using synchronous method for demonstration
        print(result)
    except Exception as e:
        print(f'An error occurred: {str(e)}')
