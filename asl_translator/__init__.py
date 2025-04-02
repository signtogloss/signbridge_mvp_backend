# asl_translator/asl_translator.py

import os
from dotenv import load_dotenv

from .base import build_prompt
from .GPTTranslator import GPTTranslator
from .MonicaTranslator import MonicaTranslator
from .DeepseekTranslator import DeepseekTranslator  # 新增引入 Deepseek

# 加载 .env 文件中的环境变量（其中包含 OPENAI_API_KEY、MONICA_API_KEY、DEEPSEEK_API_KEY 等）
load_dotenv()

# 从环境变量中读取默认的 API Key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MONICA_API_KEY = os.getenv("MONICA_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")  # 新增

def asl_translator(sentence: str, model: str = "gpt", api_key: str = None) -> str:
    """
    统一的翻译接口。
    
    参数:
        sentence (str): 需要翻译的英文句子。
        model (str): 指定使用的模型，可选 "gpt"、"monica" 或 "deepseek"，默认 "gpt"。
        api_key (str): 可选，如果不传，则从环境变量中读取默认的 Key。

    返回:
        str: 转换后的 ASL gloss，全部大写。
    """
    # 根据模型选择不同的翻译器实现
    if model.lower() == "gpt":
        key = api_key or OPENAI_API_KEY
        translator = GPTTranslator(key)
    elif model.lower() == "monica":
        key = api_key or MONICA_API_KEY
        translator = MonicaTranslator(key)
    elif model.lower() == "deepseek":
        key = api_key or DEEPSEEK_API_KEY
        translator = DeepseekTranslator(key)
        return translator.translate(sentence)
    else:
        raise ValueError(f"不支持的模型类型: {model}")

    prompt = build_prompt(sentence)
    return translator.translate(prompt)

if __name__ == "__main__":
    # 简单测试
    test_sentence = "Hello, how are you?"
    print("GPT result:", asl_translator(test_sentence, model="gpt"))
    print("Monica result:", asl_translator(test_sentence, model="monica"))
    print("Deepseek result:", asl_translator(test_sentence, model="deepseek"))