import json
import re
from typing import Dict, List, Optional
import sympy as sp
from transformers import pipeline
from scipy.spatial.distance import cosine

try:
    import requests
    from googlesearch import search
    from bs4 import BeautifulSoup

    WEB_SEARCH_AVAILABLE = True
except ImportError as e:
    print(f"Bot: Warning: Web search dependencies not installed ({e}). Web search disabled.")
    WEB_SEARCH_AVAILABLE = False

# Initialize NLP pipeline
nlp = pipeline("feature-extraction", model="distilbert-base-uncased", tokenizer="distilbert-base-uncased")


def load_knowledge_base(file_path: str) -> Dict:
    try:
        with open(file_path, 'r') as file:
            return json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        return {"questions": []}


def save_knowledge_base(file_path: str, data: Dict):
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=2)


def load_train_file(file_path: str = "Train.txt") -> List[Dict[str, str]]:
    train_data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            i = 0
            while i < len(lines):
                if lines[i].startswith("Question: ") and i + 1 < len(lines) and lines[i + 1].startswith("Answer: "):
                    question = lines[i][len("Question: "):].strip()
                    answer = lines[i + 1][len("Answer: "):].strip()
                    train_data.append({"question": question, "answer": answer})
                    i += 2
                else:
                    i += 1
        return train_data
    except FileNotFoundError:
        print(f"Bot: Warning: {file_path} not found. Skipping local text file search.")
        return []
    except Exception as e:
        print(f"Bot: Error reading {file_path}: {e}")
        return []


def find_best_match(user_question: str, questions: List[str]) -> Optional[str]:
    """Find the best matching question using sentence embeddings and cosine similarity."""
    try:
        user_embedding = nlp(user_question)[0][0]  # Get embedding for user question
        best_score, best_question = 0, None
        for q in questions:
            q_embedding = nlp(q)[0][0]  # Get embedding for each question
            score = 1 - cosine(user_embedding, q_embedding)
            if score > best_score and score > 0.85:  # Threshold for similarity
                best_score, best_question = score, q
        print(f"Debug: Best match for '{user_question}': '{best_question}' with score {best_score}")
        return best_question
    except Exception as e:
        print(f"Bot: Error in NLP matching: {e}")
        return None


def get_answer_for_question(question: str, knowledge_base: Dict) -> Optional[tuple[str, str]]:
    for q in knowledge_base["questions"]:
        if q["question"].lower() == question.lower():
            return q["answer"], q["question"]
    return None, None


def find_answer_in_train(question: str, train_data: List[Dict[str, str]]) -> Optional[tuple[str, str]]:
    for q in train_data:
        if q["question"].lower() == question.lower():
            return q["answer"], q["question"]
    questions = [q["question"] for q in train_data]
    best_match = find_best_match(question, questions)
    if best_match:
        for q in train_data:
            if q["question"].lower() == best_match.lower():
                return q["answer"], best_match
    return None, None


def fetch_web_answer(query: str, stack_overflow: bool = False) -> Optional[str]:
    if not WEB_SEARCH_AVAILABLE:
        return None
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        search_query = f"{query} site:stackoverflow.com" if stack_overflow else query
        for url in search(search_query, num_results=3):
            try:
                response = requests.get(url, headers=headers, timeout=5)
                response.raise_for_status()
                soup = BeautifulSoup(response.text, 'html.parser')
                paragraphs = soup.find_all('p')
                text = ' '.join(p.get_text() for p in paragraphs[:3] if p.get_text())
                if text:
                    return text[:500] + "..." if len(text) > 500 else text
            except requests.RequestException:
                continue
        return None
    except Exception as e:
        print(f"Bot: Error fetching web results: {e}")
        return None


def is_math_query(user_input: str) -> bool:
    """Check if the user input is a mathematical query, including natural language phrases."""
    math_keywords = [
        r'\d+\s*[\+\-\*/\^]\s*\d+',  # Basic arithmetic (e.g., 2 + 3, 5 * 4)
        r'solve\s+.*=',  # Equations (e.g., solve x^2 - 4 = 0)
        r'differentiate\s+',  # Calculus (e.g., differentiate x^2)
        r'integrate\s+',  # Calculus (e.g., integrate x^2)
        r'simplify\s+',  # Simplify expressions (e.g., simplify x^2 + 2x)
        r'=\s*\d+',  # Equations with equals (e.g., x + 2 = 5)
        r'[a-zA-Z]\s*\^\s*\d+',  # Variables with powers (e.g., x^2)
        r'(add|subtract|multiply|divide|times|plus|minus)\s+[a-zA-Z0-9]+\s+(and|plus|minus|times|by|divided by)\s+[a-zA-Z0-9]+',
        # Natural language math (e.g., add two plus two)
    ]
    return any(re.search(pattern, user_input.lower()) for pattern in math_keywords)


def parse_natural_language_math(user_input: str) -> str:
    """Convert natural language math phrases to SymPy-compatible expressions."""
    user_input = user_input.lower().strip()

    # Dictionary for word-to-number conversion
    word_to_num = {
        'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4',
        'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9'
    }

    # Dictionary for operation words to symbols
    operation_map = {
        'add': '+', 'plus': '+', 'subtract': '-', 'minus': '-',
        'multiply': '*', 'times': '*', 'divide': '/', 'divided by': '/',
        'and': '+'  # 'by' is handled separately below
    }

    # Replace word numbers with digits
    for word, num in word_to_num.items():
        user_input = re.sub(r'\b' + word + r'\b', num, user_input)

    # Detect the primary operation (first occurrence of add, multiply, etc.)
    primary_op = None
    for word, symbol in operation_map.items():
        if re.search(r'\b' + word + r'\b', user_input):
            primary_op = symbol
            user_input = re.sub(r'\b' + word + r'\b', ' ', user_input)  # Remove the operation word
            break

    # Handle connector words like 'by' or 'and'
    user_input = re.sub(r'\b(by|and|plus|minus|times|divided by)\b', ' ', user_input)

    # Clean up multiple spaces and split into parts
    parts = re.split(r'\s+', user_input.strip())
    parts = [p for p in parts if p]  # Remove empty strings

    # Construct the final expression
    if primary_op and len(parts) >= 2:
        final_expression = f"{parts[0]} {primary_op} {parts[1]}"
    else:
        final_expression = ' '.join(parts)  # Fallback to cleaned input

    return final_expression


def solve_math(user_input: str) -> Optional[str]:
    """Attempt to solve a mathematical query using SymPy, including natural language phrases."""
    try:
        x, y = sp.symbols('x y')  # Define common variables
        user_input = user_input.lower().strip()

        # Check if it's a natural language math query
        if re.search(r'(add|subtract|multiply|divide|times|plus|minus)\s+[a-zA-Z0-9]+\s+(and|plus|minus|times|by|divided by)\s+[a-zA-Z0-9]+', user_input):
            user_input = parse_natural_language_math(user_input)

        # Handle 'solve' queries (e.g., "solve x^2 - 4 = 0")
        if user_input.startswith('solve'):
            equation = user_input.replace('solve', '').strip()
            if '=' in equation:
                left, right = equation.split('=')
                eq = sp.Eq(sp.sympify(left.strip()), sp.sympify(right.strip()))
                solution = sp.solve(eq, x)
                return f"Solution: {solution}"
            else:
                return "Bot: Please provide an equation with '=' (e.g., x^2 - 4 = 0)."

        # Handle 'differentiate' queries (e.g., "differentiate x^2")
        elif user_input.startswith('differentiate'):
            expression = user_input.replace('differentiate', '').strip()
            expr = sp.sympify(expression)
            derivative = sp.diff(expr, x)
            return f"Derivative: {derivative}"

        # Handle 'integrate' queries (e.g., "integrate x^2")
        elif user_input.startswith('integrate'):
            expression = user_input.replace('integrate', '').strip()
            expr = sp.sympify(expression)
            integral = sp.integrate(expr, x)
            return f"Integral: {integral}"

        # Handle 'simplify' queries (e.g., "simplify x^2 + 2x + x")
        elif user_input.startswith('simplify'):
            expression = user_input.replace('simplify', '').strip()
            expr = sp.sympify(expression)
            simplified = sp.simplify(expr)
            return f"Simplified: {simplified}"

        # Handle direct arithmetic or expressions (e.g., "2 + 2", "x^2 + 2x")
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

def chatbot():
    knowledge_base = load_knowledge_base('knowledge_base.json')
    train_data = load_train_file('Train.txt')
    greetings = {'hi', 'hey', 'hello', 'heyy', 'hey there', 'hi there'}

    while True:
        user_input = input('You: ')
        if user_input.lower() == 'quit':
            break
        if user_input.lower() == 'reset':
            knowledge_base = {"questions": []}
            save_knowledge_base('knowledge_base.json', knowledge_base)
            print("Bot: Knowledge base reset!")
            continue
        if user_input.startswith("python -u"):
            print("Bot: Looks like a command. Please ask a question!")
            continue

        # Handle greetings
        if user_input.lower() in greetings:
            print("Bot: Hello! How can I assist you today?")
            continue

        # Check if it's a math query
        if is_math_query(user_input):
            math_result = solve_math(user_input)
            print(f"Bot: {math_result}")
            save_answer = input("Bot: Should I save this answer to the knowledge base? (yes/no): ")
            if save_answer.lower() == 'yes':
                knowledge_base["questions"].append({"question": user_input, "answer": math_result})
                save_knowledge_base('knowledge_base.json', knowledge_base)
                print("Bot: Answer saved to knowledge base!")
            continue

        # Check Train.txt first
        if train_data:
            answer, matched_question = find_answer_in_train(user_input, train_data)
            if answer:
                print(f"Bot: {answer} (from Train.txt, matched: '{matched_question}')")
                continue

        # Check local JSON knowledge base
        answer, matched_question = get_answer_for_question(user_input, knowledge_base)
        if answer:
            print(f"Bot: {answer} (from knowledge base, matched: '{matched_question}')")
            continue
        best_match = find_best_match(user_input, [q['question'] for q in knowledge_base['questions']])
        if best_match:
            answer, matched_question = get_answer_for_question(best_match, knowledge_base)
            print(f"Bot: {answer} (from knowledge base, matched: '{best_match}')")
            continue

        # Prompt to teach or search online
        print(
            "Bot: I couldn't find an answer locally. Would you like to teach me the answer or search online? (teach/search)")
        choice = input("Your choice: ").lower()
        if choice == 'teach':
            teach_answer = input("Type the answer or 'quit' to skip: ")
            if teach_answer.lower() != 'quit' and teach_answer.strip():
                knowledge_base["questions"].append({"question": user_input, "answer": teach_answer})
                save_knowledge_base('knowledge_base.json', knowledge_base)
                print("Bot: Thank you! I learned a new response!")
            else:
                print("Bot: Answer skipped, no changes made.")
            continue

        # Try general web search
        if choice == 'search':
            print("Bot: Searching the web...")
            web_answer = fetch_web_answer(user_input, stack_overflow=False)
            if web_answer:
                print(f"Bot: {web_answer} (from general web)")
                save_answer = input("Bot: Should I save this answer to the knowledge base? (yes/no): ")
                if save_answer.lower() == 'yes':
                    knowledge_base["questions"].append({"question": user_input, "answer": web_answer})
                    save_knowledge_base('knowledge_base.json', knowledge_base)
                    print("Bot: Answer saved to knowledge base!")
                continue

            # Try Stack Overflow search
            print("Bot: No general web answer found. Searching Stack Overflow...")
            so_answer = fetch_web_answer(user_input, stack_overflow=True)
            if so_answer:
                print(f"Bot: {so_answer} (from Stack Overflow)")
                save_answer = input("Bot: Should I save this answer to the knowledge base? (yes/no): ")
                if save_answer.lower() == 'yes':
                    knowledge_base["questions"].append({"question": user_input, "answer": so_answer})
                    save_knowledge_base('knowledge_base.json', knowledge_base)
                    print("Bot: Answer saved to knowledge base!")
                continue

        # Prompt to teach if no answer found
        print("Bot: I couldn't find an answer online. Can you teach me?")
        similar_questions = find_best_match(user_input, [q['question'] for q in
                                                         knowledge_base['questions']])  # Use NLP for similar questions
        if similar_questions:
            print("Bot: Did you mean this question?")
            print(f"1. {similar_questions}")
            choice = input("Enter '1' to select or 'new' to add a new question: ")
            if choice == '1':
                answer, matched_question = get_answer_for_question(similar_questions, knowledge_base)
                print(f"Bot: {answer} (from knowledge base, matched: '{similar_questions}')")
                continue

        new_answer = input("Type the answer or 'quit' to skip: ")
        if new_answer.lower() != 'quit' and new_answer.strip():
            knowledge_base["questions"].append({"question": user_input, "answer": new_answer})
            save_knowledge_base('knowledge_base.json', knowledge_base)
            print("Bot: Thank you! I learned a new response!")
        else:
            print("Bot: Answer skipped, no changes made.")


if __name__ == "__main__":
    chatbot()