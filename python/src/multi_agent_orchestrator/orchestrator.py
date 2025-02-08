# Formatting Consistency
def example_function(a, b):
    result = a + b  # Correctly spaced around the operator
    return result  # Consistent spacing around the return keyword

# Error Logging
try:
    # some code that might raise an error
except Exception as error:
    print(f"Error message: {str(error)}")  # Consistent error logging format

# String Continuation
long_string = (
    "This is a long string that uses a backslash for line continuation. "
    "It helps maintain readability."
)

# Method Definitions
class MyClass:
    def method_one(self, arg1):
        # method implementation
        pass

    def method_two(self, arg1, arg2):
        # method implementation
        pass

# Comments and Documentation
# This function adds two numbers
def add_numbers(num1, num2):
    """Adds two numbers and returns the result."""
    return num1 + num2

# Return Types and Annotations
def calculate_sum(a: int, b: int) -> int:
    return a + b

# Functionality Completeness
class Calculator:
    def add(self, a: int, b: int) -> int:
        """Adds two numbers and returns the result."""
        return a + b

    def subtract(self, a: int, b: int) -> int:
        """Subtracts b from a and returns the result."""
        return a - b