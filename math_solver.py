import re
import sympy as sp
from typing import Optional

class MathSolver:
    """Handles detection and solving of mathematical queries, including natural language."""
    
    def __init__(self):
        self.math_keywords = [
            r'\d+\s*[\+\-\*/\^]\s*\d+',  # Basic arithmetic
            r'solve\s+.*=',  # Equations
            r'differentiate\s+',  # Calculus
            r'integrate\s+',  # Calculus
            r'simplify\s+',  # Simplify expressions
            r'=\s*\d+',  # Equations with equals
            r'[a-zA-Z]\s*\^\s*\d+',  # Variables with powers
            r'(add|subtract|multiply|divide|times|plus|minus)\s+[a-zA-Z0-9]+\s+(and|plus|minus|times|by|divided by)\s+[a-zA-Z0-9]+',
        ]

    def is_math_query(self, user_input: str) -> bool:
        """Check if the input is a mathematical query."""
        return any(re.search(pattern, user_input.lower()) for pattern in self.math_keywords)

    def parse_natural_language_math(self, user_input: str) -> str:
        """Convert natural language math phrases to SymPy-compatible expressions."""
        user_input = user_input.lower().strip()
        word_to_num = {
            'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4',
            'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9'
        }
        operation_map = {
            'add': '+', 'plus': '+', 'subtract': '-', 'minus': '-',
            'multiply': '*', 'times': '*', 'divide': '/', 'divided by': '/',
            'and': '+'  # 'by' is handled separately
        }
        for word, num in word_to_num.items():
            user_input = re.sub(r'\b' + word + r'\b', num, user_input)
        primary_op = None
        for word, symbol in operation_map.items():
            if re.search(r'\b' + word + r'\b', user_input):
                primary_op = symbol
                user_input = re.sub(r'\b' + word + r'\b', ' ', user_input)
                break
        user_input = re.sub(r'\b(by|and|plus|minus|times|divided by)\b', ' ', user_input)
        parts = re.split(r'\s+', user_input.strip())
        parts = [p for p in parts if p]
        if primary_op and len(parts) >= 2:
            final_expression = f"{parts[0]} {primary_op} {parts[1]}"
        else:
            final_expression = ' '.join(parts)
        return final_expression

    def solve_math(self, user_input: str) -> Optional[str]:
        """Solve a mathematical query using SymPy."""
        try:
            x, y = sp.symbols('x y')
            user_input = user_input.lower().strip()
            if re.search(r'(add|subtract|multiply|divide|times|plus|minus)\s+[a-zA-Z0-9]+\s+(and|plus|minus|times|by|divided by)\s+[a-zA-Z0-9]+', user_input):
                user_input = self.parse_natural_language_math(user_input)
            if user_input.startswith('solve'):
                equation = user_input.replace('solve', '').strip()
                if '=' in equation:
                    left, right = equation.split('=')
                    eq = sp.Eq(sp.sympify(left.strip()), sp.sympify(right.strip()))
                    solution = sp.solve(eq, x)
                    return f"Solution: {solution}"
                else:
                    return "Bot: Please provide an equation with '=' (e.g., x^2 - 4 = 0)."
            elif user_input.startswith('differentiate'):
                expression = user_input.replace('differentiate', '').strip()
                expr = sp.sympify(expression)
                derivative = sp.diff(expr, x)
                return f"Derivative: {derivative}"
            elif user_input.startswith('integrate'):
                expression = user_input.replace('integrate', '').strip()
                expr = sp.sympify(expression)
                integral = sp.integrate(expr, x)
                return f"Integral: {integral}"
            elif user_input.startswith('simplify'):
                expression = user_input.replace('simplify', '').strip()
                expr = sp.sympify(expression)
                simplified = sp.simplify(expr)
                return f"Simplified: {simplified}"
            else:
                expr = sp.sympify(user_input)
                if expr.is_number:
                    result = expr.evalf()
                    if result.is_integer:
                        return f"Result: {int(result)}"
                    else:
                        return f"Result: {float(result):.4f}".rstrip('0').rstrip('.')
                else:
                    result = sp.simplify(expr)
                    return f"Result: {result}"
        except sp.SympifyError:
            return "Bot: Invalid mathematical expression. Please check your input."
        except Exception as e:
            return f"Bot: Error solving math problem: {str(e)}"