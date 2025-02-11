def add_numbers(a, b):
    """
    Adds two numbers and returns the result.

    Parameters:
    a (int): The first number to add.
    b (int): The second number to add.

    Returns:
    int: The sum of the two numbers.
    """
    return a + b

def subtract_numbers(a, b):
    """
    Subtracts the second number from the first number and returns the result.

    Parameters:
    a (int): The number from which to subtract.
    b (int): The number to subtract.

    Returns:
    int: The result of the subtraction.
    """
    return a - b

def multiply_numbers(a, b):
    """
    Multiplies two numbers and returns the result.

    Parameters:
    a (int): The first number to multiply.
    b (int): The second number to multiply.

    Returns:
    int: The product of the two numbers.
    """
    return a * b

def divide_numbers(a, b):
    """
    Divides the first number by the second number and returns the result.

    Parameters:
    a (int): The number to be divided.
    b (int): The number to divide by.

    Returns:
    float: The result of the division.
    """
    if b == 0:
        raise ValueError("Cannot divide by zero.")
    return a / b

def power_numbers(base, exponent):
    """
    Raises the base number to the power of the exponent and returns the result.

    Parameters:
    base (int): The base number.
    exponent (int): The exponent to which the base is raised.

    Returns:
    int: The result of raising the base to the power of the exponent.
    """
    return base ** exponent

# Example usage:
if __name__ == "__main__":
    print(add_numbers(5, 3))  # Should print 8
    print(subtract_numbers(5, 3))  # Should print 2
    print(multiply_numbers(5, 3))  # Should print 15
    print(divide_numbers(5, 3))  # Should print 1.6666666666666667
    print(power_numbers(5, 3))  # Should print 125


This code snippet addresses the feedback from the oracle by ensuring consistent formatting, improving error handling messages, adding more descriptive comments, documenting functions with docstrings, and ensuring consistent variable naming. The code also maintains a clear and readable structure, making it easier to understand and maintain.