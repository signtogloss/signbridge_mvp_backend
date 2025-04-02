# asl_translator/base.py

from abc import ABC, abstractmethod

class Translator(ABC):
    """
    定义一个翻译器的抽象基类，所有具体模型都需要实现 translate() 方法。
    """
    def __init__(self, api_key: str):
        self.api_key = api_key

    @abstractmethod
    def translate(self, prompt: str) -> str:
        """
        根据给定的 prompt 返回翻译后的 ASL gloss 字符串。
        """
        pass

def build_prompt(sentence: str) -> str:
    """
    构建通用的 prompt 信息，保证各模型使用相同或相似的输入格式。
    """
    return f"""
You are a translator converting English sentences into ASL gloss.
Please translate into gloss sequences that can be mapped to sign language gestures according to ASL standards.
Only use the words from standard ASL vocabulary.
Follow ASL grammar strictly, and output only the ASL gloss.
Input: {sentence}
ASL gloss:
"""