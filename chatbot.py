# main.py (or chatbot.py)
import os # <-- Ensure os is imported
from dotenv import load_dotenv # <-- New import

# Load environment variables from .env file
load_dotenv() # <-- New line: Call this at the very beginning

# --- Existing imports follow ---
import asyncio
from typing import Optional
import torch
import re
import warnings
import os # <-- Add this import for os.getenv

# Suppress specific UserWarnings from transformers library
warnings.filterwarnings(
    "ignore",
    message="`resume_download` is deprecated and will be removed in version 4.38. Use `force_download=True` instead.",
    category=UserWarning
)
warnings.filterwarnings(
    "ignore",
    message="Some weights of the model checkpoint at xlm-roberta-base were not used when initializing XLMRobertaForSequenceClassification: ['classifier.dense.bias', 'classifier.dense.weight', 'classifier.out_proj.bias', 'classifier.out_proj.weight']\nYou should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.",
    category=UserWarning
)
warnings.filterwarnings(
    "ignore",
    message="The `padding_side` argument has been deprecated and will be removed in v4.32. Please set `tokenizer.padding_side` instead.",
    category=UserWarning
)


# --- Re-importing necessary modules from your original separate files ---
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, AutoModelForSequenceClassification, AutoModelForSeq2SeqLM
    from deep_translator import GoogleTranslator
    import nltk
    # Directly download stopwords here, and then import
    try:
        nltk.data.find('corpora/stopwords/bengali')
    except LookupError:
        print("Downloading NLTK Bengali stopwords...")
        nltk.download('stopwords')
    from nltk.corpus import stopwords # Import after ensuring download
except ImportError as e:
    print(f"Error importing necessary libraries: {e}")
    print("Please ensure `transformers`, `torch`, `deep_translator`, and `nltk` are installed.")
    # Exit or handle gracefully if critical libraries are missing
    exit()

# Assuming these are your custom modules. If they are in subdirectories, adjust imports.
# IMPORTANT: Ensure these files (knowledge_base_manager.py, train_data_manager.py, etc.)
# exist in the same directory or are accessible via Python path.
try:
    from knowledge_base_manager import KnowledgeBaseManager
    from train_data_manager import TrainDataManager
    from nlp_processor import NLPProcessor
    from web_searcher import WebSearcher
    from math_solver import MathSolver
    from translator import Translator
    from bangla_summarizer import BanglaSummarizer
    from bangla_generator import BanglaGenerator
    from hindi_generator import HindiGenerator # <-- NEW: Import HindiGenerator
except ImportError as e:
    print(f"Error importing custom modules: {e}")
    print("Please ensure all custom module files are present in the same directory or accessible.")
    # If these are critical, you might want to exit here, or fall back to dummy implementations if possible.
    # For now, if running the __main__ block, the dummy classes will be used.


# --- Consolidated BanglaUnderstanding Class (from your second code snippet) ---
class BanglaUnderstanding:
    """
    Handles understanding and intent recognition for Bengali text.
    Uses a pre-trained XLM-RoBERTa model for sequence classification.
    """
    # ADDED: bangla_generator parameter to the constructor
    def __init__(self, model_name="xlm-roberta-base", num_labels=2, bangla_generator=None): # <--- MODIFIED
        self.tokenizer = None
        self.model = None
        self.stop_words = set(stopwords.words('bengali'))

        # Store the passed bangla_generator instance
        self.bangla_generator = bangla_generator # <--- MODIFIED

        print(f"Device set to use { 'cuda' if torch.cuda.is_available() else 'cpu' }")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
            print("BanglaUnderstanding Tokenizer and model loaded successfully.")
        except Exception as e:
            print(f"Error loading BanglaUnderstanding model {model_name}: {e}")
            print("Please ensure internet/connectivity and that all necessary libraries are installed.")
            self.tokenizer = None
            self.model = None

    def _preprocess_text(self, text):
        """Basic text preprocessing for Bengali."""
        text = text.lower()
        # Remove punctuation
        text = re.sub(r'[^\w\s]', '', text)
        # Remove numbers
        text = re.sub(r'\d+', '', text)
        # Remove extra spaces
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def _remove_stopwords(self, text):
        """Removes Bengali stopwords from text."""
        words = text.split()
        filtered_words = [word for word in words if word not in self.stop_words]
        return ' '.join(filtered_words)

    async def understand_bangla(self, text: str) -> str:
        """
        Processes Bengali text, performs intent recognition, and returns a processed string.
        (Placeholder for actual intent recognition logic)
        """
        if not self.tokenizer or not self.model:
            return "বাংলা বোঝার মডেল লোড হয়নি।" # Bangla understanding model not loaded.

        processed_text = self._preprocess_text(text)
        processed_text = self._remove_stopwords(processed_text)

        # In a real scenario, you'd use self.model to predict intent.
        # For now, we'll just return the processed text.
        return processed_text

    async def generate_response_for_bangla(self, processed_data: dict, original_text: str) -> str:
        """
        Generates a response for Bangla input using BanglaGenerator.
        """
        print(f"Generating Bangla response for: {original_text}")
        try:
            # Use the already initialized self.bangla_generator
            if self.bangla_generator:
                # Use max_new_tokens instead of max_length
                generated_text = await self.bangla_generator.generate_bangla_text(original_text, max_new_tokens=50) # <--- MODIFIED
                if generated_text and generated_text.strip() != original_text.strip(): # Avoid trivial self-repetition
                    return generated_text
                else:
                    return f"আমি আপনার বাংলা প্রশ্নটি বুঝেছি: '{original_text}'। আরও কিছু জানতে চান?" # I understand your Bangla question: '{original_text}'. Do you want to know more?
            else:
                return f"আমি আপনার বাংলা প্রশ্নটি বুঝেছি: '{original_text}'। কিভাবে সাহায্য করতে পারি?" # Fallback if generator not available
        except Exception as e:
            print(f"Error generating Bangla response with BanglaGenerator: {e}")
            return f"আমি আপনার বাংলা প্রশ্নটি বুঝেছি: '{original_text}'। কিভাবে সাহায্য করতে পারি?" # I understand your Bangla question: '{original_text}'. How can I help?


# --- Consolidated Chatbot Class ---
class Chatbot:
    """Main chatbot class that orchestrates question answering and learning."""

    def __init__(self):
        # Initialize custom managers and processors
        self.knowledge_base_manager = KnowledgeBaseManager()
        self.train_data_manager = TrainDataManager()
        self.nlp_processor = NLPProcessor()
        self.web_searcher = WebSearcher()
        self.math_solver = MathSolver()
        self.greetings = {'hi', 'hey', 'hello', 'heyy', 'hey there', 'hi there'}

        # Initialize Translator (from first snippet, uses deep_translator)
        self.translator = Translator()
        # Initialize Google Translator (for internal use by BanglaUnderstanding/translation)
        try:
            self.google_translator = GoogleTranslator(source='auto', target='en')
            print("Google Translate initialized successfully using deep_translator.")
        except Exception as e:
            print(f"Error initializing Google Translator: {e}")
            self.google_translator = None

        # DialoGPT specific initializations
        self.dialogpt_tokenizer = None
        self.dialogpt_model = None
        self.dialogpt_chat_history_ids = None
        print("Bot: Loading DialoGPT model (first time might download ~500MB)...")
        self._load_dialogpt_model()
        print("Bot: DialoGPT model loaded successfully.")

        # Initialize Bangla Summarizer
        self.bangla_summarizer = BanglaSummarizer()

        # Initialize Bangla Text Generation model
        self.bangla_generator = BanglaGenerator() # Keep this initialization here

        # Initialize BanglaUnderstanding, passing the already loaded BanglaGenerator
        # This prevents BanglaUnderstanding from re-loading the BanglaGenerator
        self.bangla_understanding = BanglaUnderstanding(bangla_generator=self.bangla_generator) # <--- MODIFIED

        # NEW: Initialize Hindi Text Generation model
        # You should set your GEMINI_API_KEY as an environment variable (e.g., in your .env file or system settings)
        self.hindi_generator = HindiGenerator() # <-- NEW: Initialize HindiGenerator

    def _load_dialogpt_model(self):
        """Loads the DialoGPT tokenizer and model."""
        try:
            self.dialogpt_tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
            self.dialogpt_model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")
        except Exception as e:
            print(f"Error loading DialoGPT model: {e}")
            print("DialoGPT will not be available. Please check internet connection or re-run `pip install transformers torch`.")
            self.dialogpt_tokenizer = None
            self.dialogpt_model = None

    def run(self):
        """Run the chatbot loop."""
        asyncio.run(self._async_run())

    async def _async_run(self):
        """Asynchronous main loop for the chatbot."""
        print("Chatbot is ready. Type 'exit' to quit.")
        while True:
            user_input = input('You: ')
            if user_input.lower() == 'exit':
                print("Bot: Goodbye!")
                break

            response = await self.process_input(user_input)
            print(f"Bot: {response}")

    def _is_bangla(self, text: str) -> bool:
        """
        Simple heuristic to detect if the text contains Bangla characters.
        Checks for characters within the common Bangla Unicode ranges.
        """
        for char in text:
            if '\u0980' <= char <= '\u09FF':
                return True
        return False

    # NEW: Hindi language detection
    def _is_hindi(self, text: str) -> bool:
        """
        Simple heuristic to detect if the text contains Hindi characters.
        Checks for characters within the Devanagari Unicode ranges.
        """
        for char in text:
            if '\u0900' <= char <= '\u097F': # Devanagari Unicode Block
                return True
        return False

    async def _generate_dialogpt_response(self, user_input: str) -> str:
        """
        Generates a conversational response using DialoGPT.
        This method will be called if no other handlers can provide an answer.
        """
        if self.dialogpt_model is None or self.dialogpt_tokenizer is None:
            return "Sorry, my conversational ability is offline."

        print("Bot: Generating a conversational response (this may take a moment)...")

        new_input_ids = self.dialogpt_tokenizer.encode(user_input + self.dialogpt_tokenizer.eos_token, return_tensors='pt')

        if self.dialogpt_chat_history_ids is None:
            bot_input_ids = new_input_ids
        else:
            bot_input_ids = torch.cat([self.dialogpt_chat_history_ids, new_input_ids], dim=-1)

        generated_ids = await asyncio.to_thread(
            self.dialogpt_model.generate,
            bot_input_ids,
            max_length=100,
            pad_token_id=self.dialogpt_tokenizer.eos_token_id,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.7,
        )

        self.dialogpt_chat_history_ids = generated_ids

        response = self.dialogpt_tokenizer.decode(generated_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
        return response

    async def process_input(self, user_input: str) -> str:
        """Process user input and return a response."""
        user_input_lower = user_input.lower()

        # 1. Handle system commands and greetings (highest priority)
        if user_input_lower == 'reset':
            self.knowledge_base_manager.reset_knowledge_base()
            return "Knowledge base reset!"
        if user_input.startswith("python -u"): # This is likely an artifact of how you run the script, not a user command
            return "Looks like a command. Please ask a question!"
        if user_input_lower in self.greetings:
            return "Hello! How can I assist you today?"

        # 2. Handle specific "Translate into Bengali" command
        translate_en_bn_prefix = "translate into bengali \""
        if user_input_lower.startswith(translate_en_bn_prefix) and user_input_lower.endswith('"'):
            sentence_to_translate = user_input[len(translate_en_bn_prefix):-1].strip()
            if sentence_to_translate:
                print("Bot: Translating English to Bengali...")
                translated_sentence = await self.translator.translate_english_to_bengali(sentence_to_translate)
                return translated_sentence
            else:
                return "অনুগ্রহ করে একটি বাক্য দিন যা অনুবাদ করতে হবে।"

        # 3. Handle specific "Translate into English" command
        translate_bn_en_prefix = "translate into english \""
        if user_input_lower.startswith(translate_bn_en_prefix) and user_input_lower.endswith('"'):
            sentence_to_translate = user_input[len(translate_bn_en_prefix):-1].strip()
            if sentence_to_translate:
                print("Bot: Translating Bengali to English...")
                translated_sentence = await self.translator.translate_bengali_to_english(sentence_to_translate)
                return translated_sentence
            else:
                return "Please provide a sentence in Bengali to translate."

        # Handle "summarize bangla" command
        summarize_bn_prefix = "summarize bangla \""
        if user_input_lower.startswith(summarize_bn_prefix) and user_input_lower.endswith('"'):
            text_to_summarize = user_input[len(summarize_bn_prefix):-1].strip()
            if text_to_summarize:
                print("Bot: Summarizing Bangla text (this may take a moment)...")
                summarized_text = await self.bangla_summarizer.summarize_bangla_text(text_to_summarize)
                return summarized_text
            else:
                return "অনুগ্রহ করে একটি বাংলা পাঠ দিন যা সারসংক্ষেপ করতে হবে।"

        # New: Bangla Text Generation Feature
        if user_input_lower.startswith("bangla_gen:"):
            prompt = user_input[len("bangla_gen:"):].strip()
            if prompt:
                return await self.bangla_generator.generate_bangla_text(prompt, max_new_tokens=50)
            else:
                return "অনুগ্রহ করে বাংলা টেক্সট জেনারেশনের জন্য একটি প্রম্পট দিন। (উদাহরণ: bangla_gen: আমি একজন)"
            
        # NEW: Hindi Text Generation Feature
        if user_input_lower.startswith("hindi_gen:"): # New command for Hindi generation
            prompt = user_input[len("hindi_gen:"):].strip()
            if prompt:
                return await self.hindi_generator.generate_hindi_text(prompt, max_output_tokens=50) # Use max_output_tokens for Gemini
            else:
                return "कृपया हिंदी टेक्स्ट जनरेशन के लिए एक प्रॉम्प्ट दें। (उदाहरण: hindi_gen: नमस्ते, आप कैसे हैं?)" # Please provide a prompt for Hindi text generation.

        # 4. Handle general Bangla queries - prioritize before English NLP/DialoGPT if primarily Bangla
        if self._is_bangla(user_input):
            print("Bot: Detected Bangla input. Processing with BanglaUnderstanding...")
            processed_data = await self.bangla_understanding.understand_bangla(user_input)
            bangla_response = await self.bangla_understanding.generate_response_for_bangla(processed_data, user_input)
            return bangla_response
        
        # NEW: Handle general Hindi queries - prioritize before English NLP/DialoGPT if primarily Hindi
        if self._is_hindi(user_input): # New: Check for Hindi input
            print("Bot: Detected Hindi input. Generating response with HindiGenerator...")
            hindi_response = await self.hindi_generator.generate_hindi_text(user_input, max_output_tokens=70) # Adjust tokens as needed
            return hindi_response
        
        else: # If not detected as Bangla or Hindi, but might contain some Bengali/Hindi or need translation for English models
            if self.google_translator:
                # This check ensures we don't translate purely English input unnecessarily
                if any('\u0980' <= char <= '\u09FF' for char in user_input) or any('\u0900' <= char <= '\u097F' for char in user_input): # Check for Bangla OR Hindi
                    print("Bot: Mixed/Potentially Bengali or Hindi input detected, attempting translation to English for broader processing.")
                    translated_input = await asyncio.to_thread(self.google_translator.translate, user_input)
                    print(f"Bot: Translated to English: {translated_input}")
                    user_input = translated_input # Update user_input for subsequent English processing


        # 5. Handle math queries
        if self.math_solver.is_math_query(user_input):
            math_result = self.math_solver.solve_math(user_input)
            save_answer = input("Bot: Should I save this answer to the knowledge base? (yes/no): ")
            if save_answer.lower() == 'yes':
                self.knowledge_base_manager.add_Youtube(user_input, math_result)
                return f"{math_result}\nAnswer saved to knowledge base!"
            return math_result

        # 6. Check Train.txt
        if self.train_data_manager.train_data:
            answer, matched_question = self.train_data_manager.find_answer_in_train(user_input, self.nlp_processor)
            if answer:
                return f"{answer} (from Train.txt, matched: '{matched_question}')"

        # 7. Check knowledge base for direct match
        answer, matched_question = self.knowledge_base_manager.get_answer_for_question(user_input)
        if answer:
            return f"{answer} (from knowledge base, matched: '{matched_question}')"

        # Fallback to NLP best match with a similarity threshold
        BEST_MATCH_THRESHOLD = 0.985

        nlp_result = self.nlp_processor.find_best_match(
            user_input,
            [q['question'] for q in self.knowledge_base_manager.knowledge_base['questions']]
        )

        best_match_question = None
        best_match_score = 0.0

        if nlp_result and isinstance(nlp_result, tuple) and len(nlp_result) == 2:
            best_match_question, best_match_score = nlp_result
            print(f"Debug: NLP best match found: '{best_match_question}' with score {best_match_score}")

        if best_match_question and best_match_score >= BEST_MATCH_THRESHOLD:
            answer, matched_question = self.knowledge_base_manager.get_answer_for_question(best_match_question)
            if answer:
                return f"{answer} (from knowledge base, matched: '{matched_question}')"

        # Fallback to DialoGPT for general conversation if no specific answer found
        if self.dialogpt_model is not None:
            return await self._generate_dialogpt_response(user_input)

        # 9. If no answer found, prompt to teach or search
        choice = input(
            "Bot: I couldn't find an answer locally. Would you like to teach me the answer or search online? (teach/search): ").lower()
        if choice == 'teach':
            teach_answer = input("Type the answer or 'quit' to skip: ")
            if teach_answer.lower() != 'quit' and teach_answer.strip():
                self.knowledge_base_manager.add_Youtube(user_input, teach_answer)
                return "Thank you! I learned a new response!"
            return "Answer skipped, no changes made."

        # Web search
        if choice == 'search':
            print("Bot: Searching the web...")
            web_answer = self.web_searcher.fetch_web_answer(user_input, stack_overflow=False)
            if web_answer:
                save_answer = input("Bot: Should I save this answer to the knowledge base? (yes/no): ")
                if save_answer.lower() == 'yes':
                    self.knowledge_base_manager.add_Youtube(user_input, web_answer)
                    return f"{web_answer} (from general web)\nAnswer saved to knowledge base!"
                return f"{web_answer} (from general web)"

            print("Bot: No general web answer found. Searching Stack Overflow...")
            so_answer = self.web_searcher.fetch_web_answer(user_input, stack_overflow=True)
            if so_answer:
                save_answer = input("Bot: Should I save this answer to the knowledge base? (yes/no): ")
                if save_answer.lower() == 'yes':
                    self.knowledge_base_manager.add_Youtube(user_input, so_answer)
                    return f"{so_answer} (from Stack Overflow)\nAnswer saved to knowledge base!"
                return f"{so_answer} (from Stack Overflow)"

        similar_questions_for_teaching = self.nlp_processor.find_best_match(user_input, [q['question'] for q in
                                                                                           self.knowledge_base_manager.knowledge_base[
                                                                                               'questions']])
        if similar_questions_for_teaching and isinstance(similar_questions_for_teaching, tuple):
            similar_question_text, _ = similar_questions_for_teaching

            print("Bot: Did you mean this question?")
            print(f"1. {similar_question_text}")
            choice = input("Enter '1' to select or 'new' to add a new question: ")
            if choice == '1':
                answer, matched_question = self.knowledge_base_manager.get_answer_for_question(similar_question_text)
                return f"{answer} (from knowledge base, matched: '{similar_question_text}')"

        new_answer = input("Type the answer or 'quit' to skip: ")
        if new_answer.lower() != 'quit' and new_answer.strip():
            self.knowledge_base_manager.add_Youtube(user_input, new_answer)
            return "Thank you! I learned a new response!"
        return "Answer skipped, no changes made."


# --- Main execution entry point (retained with dummy implementations for self-contained testing) ---
if __name__ == '__main__':
    # Dummy implementations (keep as is if you are running chatbot.py directly without
    # the actual separate module files)
    class KnowledgeBaseManager:
        def __init__(self):
            self.knowledge_base = {'questions': []}

        def add_Youtube(self, q, a):
            self.knowledge_base['questions'].append({'question': q, 'answer': a})

        def get_answer_for_question(self, q):
            for entry in self.knowledge_base['questions']:
                if entry['question'] == q:
                    return entry['answer'], q
            return None, None

        def reset_knowledge(self):
            self.knowledge_base = {'questions': []}

    class TrainDataManager:
        def __init__(self):
            self.train_data = []

        def find_answer_in_train(self, q, nlp_processor):
            for item in self.train_data:
                if q.lower() == item[0].lower():
                    return item[1], item[0]
            return None, None

    class NLPProcessor:
        def find_best_match(self, q, questions):
            for existing_q in questions:
                if q.lower() == existing_q.lower():
                    if "who are you" in existing_q.lower():
                            return existing_q, 1.0
                    return existing_q, 1.0
            if "how is" in q.lower() or "life going" in q.lower() or "favorite color" in q.lower() or "what can you do" in q.lower():
                return "who are you?", 0.975 # Simulate a high but below threshold score
            return None, 0.0

    class WebSearcher:
        def fetch_web_answer(self, q, stack_overflow=False):
            if "weather" in q.lower():
                return "The weather is sunny."
            if "python" in q.lower() and stack_overflow:
                return "Python is a popular programming language. (from SO)"
            return None

    class MathSolver:
        def is_math_query(self, q):
            return any(op in q for op in ['+', '-', '*', '/', '^'])

        def solve_math(self, q):
            try:
                return str(eval(q.replace('^', '**')))
            except:
                return "Could not solve the mathematical expression."

    class Translator:
        def __init__(self):
            print("Dummy Translator initialized.")
        async def translate_english_to_bengali(self, text: str) -> str:
            print(f"Dummy translating English to Bengali: {text}")
            return f"ডামি অনুবাদ: {text}"
        async def translate_bengali_to_english(self, text: str) -> str:
            print(f"Dummy translating Bengali to English: {text}")
            return f"Dummy Translation: {text}"

    class BanglaSummarizer:
        def __init__(self, model_name: str = "dummy/path"):
            self.model_name = model_name
            self.summarizer_pipeline = True # Simulate loaded state
            print("Dummy Bangla Summarizer loaded.")

        async def summarize_bangla_text(self, text: str) -> str:
            if not self.summarizer_pipeline:
                return "Dummy Error: Summarization model is not loaded."
            print(f"Dummy Summarizing: {text}")
            return f"ডামি সারসংক্ষেপ: '{text[:30]}...'" # Dummy Bengali summary

    class BanglaGenerator:
        def __init__(self):
            print("Dummy BanglaGenerator initialized.")
        # IMPORTANT: Change max_length to max_new_tokens for consistency with actual model
        async def generate_bangla_text(self, prompt: str, max_new_tokens: int = 50) -> str: # <--- MODIFIED DUMMY
            print(f"Dummy generating Bangla text for prompt: {prompt}")
            return f"ডামি বাংলা জেনারেশন: {prompt} থেকে কিছু টেক্সট।"

    # NEW DUMMY: HindiGenerator
    class HindiGenerator:
        def __init__(self, api_key: str = ""):
            print("Dummy HindiGenerator initialized.")
            self.api_key = api_key # Store dummy key for consistency

        async def generate_hindi_text(self, user_input_hindi: str, max_output_tokens: int = 200, temperature: float = 0.7) -> str:
            print(f"Dummy generating Hindi text for prompt: {user_input_hindi}")
            return f"डम्मी हिंदी जनरेशन: {user_input_hindi} से कुछ टेक्स्ट।" # Dummy Hindi generation

    # If your actual modules are not found, these dummy ones will be used.
    # If they are found, the 'try...except ImportError' blocks at the top will succeed.
    try:
        from knowledge_base_manager import KnowledgeBaseManager
        from train_data_manager import TrainDataManager
        from nlp_processor import NLPProcessor
        from web_searcher import WebSearcher
        from math_solver import MathSolver
        from translator import Translator
        from bangla_summarizer import BanglaSummarizer
        from bangla_generator import BanglaGenerator
        from hindi_generator import HindiGenerator # <-- NEW: Import actual HindiGenerator
    except ImportError as e:
        print(f"One or more custom modules not found. Using dummy implementations for testing. Error: {e}")
        # Assign dummy classes to the module names so the Chatbot can still initialize
        # This part ensures that if you run `main.py` which imports `chatbot`,
        # and if the real modules are missing, the dummy ones in `chatbot.py`'s `__main__`
        # block are used for a graceful fallback.

    chatbot = Chatbot()
    chatbot.run()