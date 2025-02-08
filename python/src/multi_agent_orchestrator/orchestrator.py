import logging
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

@dataclass
class Configuration:
    param1: str
    param2: int

class MyClass:
    def __init__(self, config: Configuration):
        self.config = config

    def process_data(self, data: list) -> list:
        '''Processes the input data and returns the processed data.'''        
        try:
            processed_data = [item * 2 for item in data]  # Example processing
            return processed_data
        except Exception as e:
            logger.error(f"Error processing data: {str(e)}")
            raise

# Example usage
if __name__ == "__main__":
    config = Configuration(param1="example", param2=123)
    instance = MyClass(config)
    data = [1, 2, 3, 4]
    try:
        result = instance.process_data(data)
        print(result)
    except Exception as e:
        print(f"An error occurred: {str(e)}")
