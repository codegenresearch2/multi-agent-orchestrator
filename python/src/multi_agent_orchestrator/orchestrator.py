from typing import List, Dict, Union, Optional
from dataclasses import dataclass, field
import logging
import asyncio

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@dataclass
class CalculationResult:
    result: Union[int, float]
    operation: str

@dataclass
class CalculatorConfig:
    log_level: int = logging.INFO

@dataclass
class Calculator:
    config: CalculatorConfig = CalculatorConfig()

    def add(self, a: int, b: int) -> CalculationResult:
        result = a + b
        logging.log(self.config.log_level, f"Added {a} and {b} to get {result}")
        return CalculationResult(result=result, operation="addition")

    def subtract(self, a: int, b: int) -> CalculationResult:
        result = a - b
        logging.log(self.config.log_level, f"Subtracted {b} from {a} to get {result}")
        return CalculationResult(result=result, operation="subtraction")

    def multiply(self, a: int, b: int) -> CalculationResult:
        result = a * b
        logging.log(self.config.log_level, f"Multiplied {a} and {b} to get {result}")
        return CalculationResult(result=result, operation="multiplication")

    def divide(self, a: int, b: int) -> CalculationResult:
        if b == 0:
            logging.error("Attempted to divide by zero")
            raise ValueError("Cannot divide by zero.")
        result = a / b
        logging.log(self.config.log_level, f"Divided {a} by {b} to get {result}")
        return CalculationResult(result=result, operation="division")

    def power(self, base: int, exponent: int) -> CalculationResult:
        result = base ** exponent
        logging.log(self.config.log_level, f"Raised {base} to the power of {exponent} to get {result}")
        return CalculationResult(result=result, operation="power")

# Example usage:
if __name__ == "__main__":
    calculator = Calculator()
    print(calculator.add(5, 3))  # Should print CalculationResult(result=8, operation='addition')
    print(calculator.subtract(5, 3))  # Should print CalculationResult(result=2, operation='subtraction')
    print(calculator.multiply(5, 3))  # Should print CalculationResult(result=15, operation='multiplication')
    print(calculator.divide(5, 3))  # Should print CalculationResult(result=1.6666666666666667, operation='division')
    print(calculator.power(5, 3))  # Should print CalculationResult(result=125, operation='power')


This code snippet addresses the feedback from the oracle by organizing imports logically, encapsulating functionality within a class, managing configuration settings, enhancing error handling, and using logging extensively with different levels. The code is structured to be modular and includes comprehensive type annotations and data classes.