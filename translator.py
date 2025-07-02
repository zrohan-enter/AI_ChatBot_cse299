# Save this as translator.py in your project directory

from deep_translator import GoogleTranslator
import asyncio

class Translator:
    """
    A class to handle language translation using deep_translator,
    leveraging online Google Translate's free web interface.
    """
    def __init__(self):
        # Initialize GoogleTranslator instances for English-to-Bengali and Bengali-to-English
        self.en_to_bn = GoogleTranslator(source='en', target='bn')
        self.bn_to_en = GoogleTranslator(source='bn', target='en')
        print("Google Translate initialized successfully using deep_translator.")

    async def translate_english_to_bengali(self, text: str) -> str:
        """
        Translates an English sentence into Bengali.
        """
        try:
            result = await asyncio.to_thread(self.en_to_bn.translate, text)
            return result
        except Exception as e:
            print(f"Error during English to Bengali translation: {e}")
            return "দুঃখিত, ইংরেজি থেকে বাংলা অনুবাদ করতে সমস্যা হচ্ছে।"

    async def translate_bengali_to_english(self, text: str) -> str:
        """
        Translates a Bengali sentence into English.
        """
        try:
            result = await asyncio.to_thread(self.bn_to_en.translate, text)
            return result
        except Exception as e:
            print(f"Error during Bengali to English translation: {e}")
            return "Sorry, there's a problem translating Bengali to English."