# chatbot.py
import os
from dotenv import load_dotenv
import asyncio
from typing import Optional
import torch
import re
import warnings
from transformers import AutoModelForCausalLM, AutoTokenizer, MT5ForConditionalGeneration, MT5Tokenizer, AutoModelForSequenceClassification
from deep_translator import GoogleTranslator
import nltk
try:
    nltk.data.find('corpora/stopwords/bengali')
except LookupError:
    print("Downloading NLTK Bengali stopwords...")
    nltk.download('stopwords')
from nltk.corpus import stopwords
try:
    from rental_predictor import RentalPredictor
    from knowledge_base_manager import KnowledgeBaseManager
    from train_data_manager import TrainDataManager
    from nlp_processor import NLPProcessor
    from web_searcher import WebSearcher
    from math_solver import MathSolver
    from translator import Translator
    from bangla_generator import BanglaGenerator
    from hindi_generator import HindiGenerator
except ImportError as e:
    print(f"Error importing custom modules: {e}")
    print("Using dummy implementations for missing modules.")
    class KnowledgeBaseManager:
        def __init__(self): self.knowledge_base = {'questions': []}
        def add_Youtube(self, q, a): self.knowledge_base['questions'].append({'question': q, 'answer': a})
        def get_answer_for_question(self, q): return None, None
        def reset_knowledge_base(self): self.knowledge_base = {'questions': []}
    class TrainDataManager:
        def __init__(self): self.train_data = []
        def find_answer_in_train(self, q, nlp_processor): return None, None
    class NLPProcessor:
        def find_best_match(self, q, questions): return None, 0.0
    class WebSearcher:
        def fetch_web_answer(self, q, stack_overflow=False): return None
    class MathSolver:
        def is_math_query(self, q): return False
        def solve_math(self, q): return "Could not solve."
    class Translator:
        async def translate_english_to_bengali(self, text: str) -> str: return f"Dummy Eng to Bn: {text}"
        async def translate_bengali_to_english(self, text: str) -> str: return f"Dummy Bn to Eng: {text}"
    class BanglaGenerator:
        async def generate_bangla_text(self, prompt: str, max_new_tokens: int = 50) -> str: return f"ডামি বাংলা জেনারেশন: {prompt} থেকে কিছু টেক্সট।"
    class HindiGenerator:
        async def generate_hindi_text(self, user_input_hindi: str, max_output_tokens: int = 200, temperature: float = 0.7) -> str: return f"डम्मी हिंदी जनरेशन: {user_input_hindi} से कुछ टेक्स्ट।"

warnings.filterwarnings("ignore", category=UserWarning)
load_dotenv()

class BanglaSummarizer:
    def __init__(self, model_name="tashfiq61/bengali-summarizer-mt5"):
        self.model = None
        self.tokenizer = None
        print(f"Bot: Loading Bangla Summarization model: {model_name} (first time might download ~1.2GB)...")
        try:
            self.tokenizer = MT5Tokenizer.from_pretrained(model_name)
            self.model = MT5ForConditionalGeneration.from_pretrained(model_name)
            print(f"Bot: Bangla Summarization model {model_name} loaded successfully.")
        except Exception as e:
            print(f"Error loading Bangla summarization model {model_name}: {e}")
            print("Please ensure internet/connectivity and that sentencepiece is installed.")
            print("Bot: Failed to load Bangla Summarization model. Summarization will not be available.")

    async def summarize_bangla_text(self, text: str) -> str:
        if not self.model or not self.tokenizer:
            return "ডামি সারসংক্ষেপ: বাংলা সারাংশ মডেল লোড হয়নি।"
        try:
            inputs = self.tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
            summary_ids = self.model.generate(
                inputs["input_ids"],
                max_length=150,
                min_length=30,
                length_penalty=2.0,
                num_beams=4,
                early_stopping=True
            )
            summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            return summary
        except Exception as e:
            print(f"Error summarizing text: {e}")
            return f"ডামি সারসংক্ষেপ: '{text[:30]}...'"

class BanglaUnderstanding:
    def __init__(self, model_name="xlm-roberta-base", num_labels=2, bangla_generator=None):
        self.tokenizer = None
        self.model = None
        self.stop_words = set(stopwords.words('bengali'))
        self.bangla_generator = bangla_generator
        print(f"Device set to use {'cuda' if torch.cuda.is_available() else 'cpu'}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
            print("BanglaUnderstanding Tokenizer and model loaded successfully.")
        except Exception as e:
            print(f"Error loading BanglaUnderstanding model: {e}")
            self.tokenizer = None
            self.model = None

    def _preprocess_text(self, text):
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def _remove_stopwords(self, text):
        words = text.split()
        filtered_words = [word for word in words if word not in self.stop_words]
        return ' '.join(filtered_words)

    async def understand_bangla(self, text: str) -> str:
        if not self.tokenizer or not self.model:
            return "বাংলা বোঝার মডেল লোড হয়নি।"
        processed_text = self._preprocess_text(text)
        processed_text = self._remove_stopwords(processed_text)
        return processed_text

    async def generate_response_for_bangla(self, processed_data: dict, original_text: str) -> str:
        print(f"Generating Bangla response for: {original_text}")
        try:
            if self.bangla_generator:
                generated_text = await self.bangla_generator.generate_bangla_text(original_text, max_new_tokens=50)
                if generated_text and generated_text.strip() != original_text.strip():
                    return generated_text
                return f"আমি আপনার বাংলা প্রশ্নটি বুঝেছি: '{original_text}'। আরও কিছু জানতে চান?"
            return f"আমি আপনার বাংলা প্রশ্নটি বুঝেছি: '{original_text}'। কিভাবে সাহায্য করতে পারি?"
        except Exception as e:
            print(f"Error generating Bangla response: {e}")
            return f"আমি আপনার বাংলা প্রশ্নটি বুঝেছি: '{original_text}'। কিভাবে সাহায্য করতে পারি?"

class Chatbot:
    def __init__(self):
        self.knowledge_base_manager = KnowledgeBaseManager()
        self.train_data_manager = TrainDataManager()
        self.nlp_processor = NLPProcessor()
        self.web_searcher = WebSearcher()
        self.math_solver = MathSolver()
        self.greetings = {'hi', 'hey', 'hello', 'heyy', 'hey there', 'hi there'}
        self.translator = Translator()
        try:
            self.google_translator = GoogleTranslator(source='auto', target='en')
            print("Google Translate initialized successfully.")
        except Exception as e:
            print(f"Error initializing Google Translator: {e}")
            self.google_translator = None
        self.dialogpt_tokenizer = None
        self.dialogpt_model = None
        self.dialogpt_chat_history_ids = None
        print("Bot: Loading DialoGPT model...")
        self._load_dialogpt_model()
        print("Bot: DialoGPT model loaded successfully.")
        self.bangla_summarizer = BanglaSummarizer()
        self.bangla_generator = BanglaGenerator()
        self.bangla_understanding = BanglaUnderstanding(bangla_generator=self.bangla_generator)
        self.hindi_generator = HindiGenerator()
        try:
            self.rental_predictor = RentalPredictor(model_path='rental_predictor_model.pkl', columns_path='original_X_columns.pkl')
            print("Rental Predictor initialized with pre-trained model.")
        except Exception as e:
            print(f"Error initializing RentalPredictor with pre-trained model: {e}")
            print("Falling back to training with rental_data.csv...")
            try:
                self.rental_predictor = RentalPredictor(data_path='rental_data.csv')
                print("Rental Predictor initialized with new training.")
            except Exception as e2:
                print(f"Error training RentalPredictor: {e2}")
                self.rental_predictor = None

    def _load_dialogpt_model(self):
        try:
            self.dialogpt_tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
            self.dialogpt_model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")
        except Exception as e:
            print(f"Error loading DialoGPT model: {e}")
            self.dialogpt_tokenizer = None
            self.dialogpt_model = None

    def run(self):
        asyncio.run(self._async_run())

    async def _async_run(self):
        print("Chatbot is ready. Type 'exit' to quit.")
        while True:
            user_input = input('You: ')
            if user_input.lower() == 'exit':
                print("Bot: Goodbye!")
                break
            response = await self.process_input(user_input)
            print(f"Bot: {response}")

    def _is_bangla(self, text: str) -> bool:
        for char in text:
            if '\u0980' <= char <= '\u09FF':
                return True
        return False

    def _is_hindi(self, text: str) -> bool:
        for char in text:
            if '\u0900' <= char <= '\u097F':
                return True
        return False

    def _is_rental_query(self, text: str) -> bool:
        keywords = ['predict rent', 'rent price', 'how much rent', 'rental amount for', 'flat rent']
        bangla_keywords = ['ভাড়া কত', 'ফ্ল্যাটের ভাড়া', 'ভাড়া পূর্বাভাস']
        return any(keyword in text.lower() for keyword in keywords) or any(keyword in text for keyword in bangla_keywords)

    def _parse_rental_query(self, text: str):
        original_text = text
        text_lower = text.lower()
        location, size_sqft, position, floor, rating = None, None, 'front', None, None
        if self._is_bangla(text) and self.google_translator:
            print("Bot: Translating Bangla rental query to English...")
            try:
                # Preprocess Bangla numbers to ensure correct translation
                bangla_digits = {'০': '0', '১': '1', '২': '2', '৩': '3', '৪': '4', '৫': '5', '৬': '6', '৭': '7', '৮': '8', '৯': '9'}
                for bn_digit, en_digit in bangla_digits.items():
                    text = text.replace(bn_digit, en_digit)
                text_lower = self.google_translator.translate(text)
                print(f"Translated: {text_lower}")
            except Exception as e:
                print(f"Translation error: {e}")
                text_lower = original_text.lower()
        known_locations = ['gulshan', 'banani', 'dhanmondi', 'mirpur', 'uttara', 'mohakhali', 'bashundhara', 'shyamoli', 'motijheel', 'rampura']
        # Improved regex to match locations with or without 'in'/'at'
        location_matches = re.search(r'\b(gulshan|banani|dhanmondi|mirpur|uttara|mohakhali|bashundhara|shyamoli|motijheel|rampura)\b', text_lower, re.IGNORECASE)
        if location_matches:
            location = location_matches.group(1).capitalize()
        size_matches = re.search(r'(\d+)\s*(sqft|square\s*feet|sq\s*ft)', text_lower, re.IGNORECASE)
        if size_matches:
            size_sqft = int(size_matches.group(1))
        if 'front' in text_lower:
            position = 'front'
        elif 'back' in text_lower:
            position = 'back'
        floor_matches = re.search(r'(?:(\d+)(?:st|nd|rd|th)?\s+floor|floor\s+(\d+))', text_lower, re.IGNORECASE)
        if floor_matches:
            floor = int(floor_matches.group(1) or floor_matches.group(2))
        rating_matches = re.search(r'(?:rating\s+(\d+\.?\d*)|(\d+\.?\d*)\s+rating)', text_lower, re.IGNORECASE)
        if rating_matches:
            rating = float(rating_matches.group(1) or rating_matches.group(2))
        return {
            'location': location,
            'size_sqft': size_sqft,
            'position': position,
            'floor': floor,
            'rating': rating
        }

    async def _generate_dialogpt_response(self, user_input: str) -> str:
        if self.dialogpt_model is None or self.dialogpt_tokenizer is None:
            return "Sorry, my conversational ability is offline."
        print("Bot: Generating a conversational response...")
        new_input_ids = self.dialogpt_tokenizer.encode(user_input + self.dialogpt_tokenizer.eos_token, return_tensors='pt')
        bot_input_ids = new_input_ids if self.dialogpt_chat_history_ids is None else torch.cat([self.dialogpt_chat_history_ids, new_input_ids], dim=-1)
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
        user_input_lower = user_input.lower()
        if user_input_lower == 'reset':
            self.knowledge_base_manager.reset_knowledge_base()
            return "Knowledge base reset!"
        if user_input.startswith("python -u"):
            return "Looks like a command. Please ask a question!"
        if user_input_lower in self.greetings:
            return "Hello! How can I assist you today?"
        if self._is_rental_query(user_input):
            print("Bot: Detected rental price prediction query. Parsing details...")
            rental_details = self._parse_rental_query(user_input)
            print(f"DEBUG: Parsed rental details: {rental_details}")
            required_fields = ['location', 'size_sqft']
            missing_info = [k for k in required_fields if rental_details.get(k) is None]
            if missing_info:
                return f"Sorry, I need more details to predict the rent. Missing: {', '.join(missing_info)}. Please provide location and size (e.g., '750 sqft in Gulshan')."
            if not self.rental_predictor:
                return "Sorry, rental prediction is unavailable due to initialization error."
            try:
                predicted_rent = self.rental_predictor.predict_rental_price(
                    location=rental_details['location'],
                    size_sqft=rental_details['size_sqft'],
                    position=rental_details['position'],
                    floor=rental_details['floor'],
                    rating=rental_details['rating']
                )
                return f"Based on the details provided, the predicted rental amount is approximately ${predicted_rent:.2f}."
            except ValueError as e:
                return f"Error: {e}"
        translate_en_bn_prefix = "translate into bengali \""
        if user_input_lower.startswith(translate_en_bn_prefix) and user_input_lower.endswith('"'):
            sentence = user_input[len(translate_en_bn_prefix):-1].strip()
            if sentence:
                print("Bot: Translating English to Bengali...")
                return await self.translator.translate_english_to_bengali(sentence)
            return "অনুগ্রহ করে একটি বাক্য দিন যা অনুবাদ করতে হবে।"
        translate_bn_en_prefix = "translate into english \""
        if user_input_lower.startswith(translate_bn_en_prefix) and user_input_lower.endswith('"'):
            sentence = user_input[len(translate_bn_en_prefix):-1].strip()
            if sentence:
                print("Bot: Translating Bengali to English...")
                return await self.translator.translate_bengali_to_english(sentence)
            return "Please provide a sentence in Bengali to translate."
        summarize_bn_prefix = "summarize bangla \""
        if user_input_lower.startswith(summarize_bn_prefix) and user_input_lower.endswith('"'):
            text = user_input[len(summarize_bn_prefix):-1].strip()
            if text:
                print("Bot: Summarizing Bangla text...")
                return await self.bangla_summarizer.summarize_bangla_text(text)
            return "অনুগ্রহ করে একটি বাংলা পাঠ দিন যা সারসংক্ষেপ করতে হবে।"
        if user_input_lower.startswith("bangla_gen:"):
            prompt = user_input[len("bangla_gen:"):].strip()
            if prompt:
                return await self.bangla_generator.generate_bangla_text(prompt, max_new_tokens=50)
            return "অনুগ্রহ করে বাংলা টেক্সট জেনারেশনের জন্য একটি প্রম্পট দিন।"
        if user_input_lower.startswith("hindi_gen:"):
            prompt = user_input[len("hindi_gen:"):].strip()
            if prompt:
                return await self.hindi_generator.generate_hindi_text(prompt, max_output_tokens=50)
            return "कृपया हिंदी टेक्स्ट जनरेशन के लिए एक प्रॉम्प्ट दें।"
        if self._is_bangla(user_input):
            print("Bot: Detected Bangla input...")
            processed_data = await self.bangla_understanding.understand_bangla(user_input)
            return await self.bangla_understanding.generate_response_for_bangla(processed_data, user_input)
        if self._is_hindi(user_input):
            print("Bot: Detected Hindi input...")
            return await self.hindi_generator.generate_hindi_text(user_input, max_output_tokens=70)
        if self.google_translator and (self._is_bangla(user_input) or self._is_hindi(user_input)):
            print("Bot: Translating mixed input to English...")
            translated_input = await asyncio.to_thread(self.google_translator.translate, user_input)
            print(f"Bot: Translated to English: {translated_input}")
            user_input = translated_input
        if self.math_solver.is_math_query(user_input):
            math_result = self.math_solver.solve_math(user_input)
            save_answer = input("Bot: Should I save this answer to the knowledge base? (yes/no): ")
            if save_answer.lower() == 'yes':
                self.knowledge_base_manager.add_Youtube(user_input, math_result)
                return f"{math_result}\nAnswer saved to knowledge base!"
            return math_result
        if self.train_data_manager.train_data:
            answer, matched_question = self.train_data_manager.find_answer_in_train(user_input, self.nlp_processor)
            if answer:
                return f"{answer} (from Train.txt, matched: '{matched_question}')"
        answer, matched_question = self.knowledge_base_manager.get_answer_for_question(user_input)
        if answer:
            return f"{answer} (from knowledge base, matched: '{matched_question}')"
        BEST_MATCH_THRESHOLD = 0.985
        nlp_result = self.nlp_processor.find_best_match(
            user_input, [q['question'] for q in self.knowledge_base_manager.knowledge_base['questions']]
        )
        best_match_question, best_match_score = nlp_result if nlp_result and isinstance(nlp_result, tuple) else (None, 0.0)
        if best_match_question and best_match_score >= BEST_MATCH_THRESHOLD:
            answer, matched_question = self.knowledge_base_manager.get_answer_for_question(best_match_question)
            if answer:
                return f"{answer} (from knowledge base, matched: '{matched_question}')"
        if self.dialogpt_model is not None:
            return await self._generate_dialogpt_response(user_input)
        choice = input("Bot: I couldn't find an answer locally. Would you like to teach me the answer or search online? (teach/search): ").lower()
        if choice == 'teach':
            teach_answer = input("Type the answer or 'quit' to skip: ")
            if teach_answer.lower() != 'quit' and teach_answer.strip():
                self.knowledge_base_manager.add_Youtube(user_input, teach_answer)
                return "Thank you! I learned a new response!"
            return "Answer skipped, no changes made."
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
        similar_questions = self.nlp_processor.find_best_match(user_input, [q['question'] for q in self.knowledge_base_manager.knowledge_base['questions']])
        if similar_questions and isinstance(similar_questions, tuple):
            similar_question_text, _ = similar_questions
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

if __name__ == '__main__':
    chatbot = Chatbot()
    chatbot.run()