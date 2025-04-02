# asl_translator/GPTTranslator.py

from openai import OpenAI
from .base import Translator, build_prompt

class GPTTranslator(Translator):
    """
    基于 GPT 的翻译器实现，调用 OpenAI API。
    """
    def __init__(self, api_key: str):
        super().__init__(api_key)
        self.client = OpenAI(api_key=self.api_key)
    
    def translate(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an ASL gloss translator."},
                {"role": "user", "content": prompt}
            ],
        )
        return response.choices[0].message.content.strip().upper()

def translate_asl(sentence: str, api_key: str) -> str:
    """
    一个便捷函数，用于直接传入英文句子和 API Key，
    内部构建 prompt 并调用 GPTTranslator 完成翻译。
    """
    prompt = build_prompt(sentence)
    translator = GPTTranslator(api_key)
    return translator.translate(prompt)