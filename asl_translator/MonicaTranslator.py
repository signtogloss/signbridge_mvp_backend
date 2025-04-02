# asl_translator/MonicaTranslator.py

import requests
from .base import Translator

class MonicaTranslator(Translator):
    """
    基于 Monica(Deepseek) 的翻译器实现。
    """

    def translate(self, prompt: str) -> str:
        """
        根据 prompt 调用 Monica(Deepseek) 接口，返回翻译后的 ASL gloss。
        prompt 是在上层通过 build_prompt(sentence) 生成的文本。
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        # 在官方示例中，使用了 "model": "gpt-4o"；你也可以改成官方支持的其他模型
        data = {
            "model": "gpt-4o-2024-11-20",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }
            ]
        }

        response = requests.post(
            "https://openapi.monica.im/v1/chat/completions",
            headers=headers,
            json=data
        )
        resp_json = response.json()

        # 根据官方文档，返回结构可能包含 choices -> [0] -> message -> content
        # 请确认实际返回的 JSON 结构是否一致
        return resp_json["choices"][0]["message"]["content"].strip().upper()