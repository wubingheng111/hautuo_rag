"""
DeepSeek LLM客户端
"""
import openai
from typing import List, Dict, Any, Optional
from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun

from config import settings


class DeepSeekLLM(LLM):
    """DeepSeek LLM包装器，兼容LangChain"""

    client: Any = None
    model_name: str = settings.DEEPSEEK_MODEL
    temperature: float = settings.TEMPERATURE
    max_tokens: int = settings.MAX_TOKENS

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.client = openai.OpenAI(
            api_key=settings.DEEPSEEK_API_KEY,
            base_url=settings.DEEPSEEK_BASE_URL
        )

    @property
    def _llm_type(self) -> str:
        return "deepseek"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """调用DeepSeek API"""
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stop=stop,
                **kwargs
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"DeepSeek API调用失败: {e}")
            return f"抱歉，我遇到了一些技术问题，无法回答您的问题。错误信息: {str(e)}"


class DeepSeekChatClient:
    """DeepSeek聊天客户端"""

    def __init__(self):
        self.client = openai.OpenAI(
            api_key=settings.DEEPSEEK_API_KEY,
            base_url=settings.DEEPSEEK_BASE_URL
        )
        self.model = settings.DEEPSEEK_MODEL

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = None,
        max_tokens: int = None,
        **kwargs
    ) -> str:
        """
        聊天完成

        Args:
            messages: 消息列表
            temperature: 温度参数
            max_tokens: 最大token数

        Returns:
            回复内容
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature or settings.TEMPERATURE,
                max_tokens=max_tokens or settings.MAX_TOKENS,
                **kwargs
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"DeepSeek API调用失败: {e}")
            return f"抱歉，我遇到了一些技术问题。错误: {str(e)}"

    def generate_response(
        self,
        user_message: str,
        context: str = "",
        system_prompt: str = None
    ) -> str:
        """
        生成回复

        Args:
            user_message: 用户消息
            context: 上下文信息
            system_prompt: 系统提示词

        Returns:
            生成的回复
        """
        messages = []

        # 添加系统提示词
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        # 构建用户消息
        if context:
            user_content = f"参考信息：\n{context}\n\n用户问题：{user_message}"
        else:
            user_content = user_message

        messages.append({"role": "user", "content": user_content})

        return self.chat_completion(messages)

    def generate_response_with_cot(
        self,
        user_message: str,
        context: str = "",
        system_prompt: str = None
    ) -> dict:
        """
        生成包含CoT推理过程的回复

        Args:
            user_message: 用户消息
            context: 上下文信息
            system_prompt: 系统提示词

        Returns:
            包含推理过程和最终答案的字典
        """
        try:
            # 构建CoT推理提示
            cot_system_prompt = """你是华佗医学AI助手，请使用Chain of Thought推理来回答问题。

请严格按照以下格式回答，确保包含完整的推理过程：

<thinking>
请在这里详细展示你的推理过程：
1. 问题理解：分析用户问题的核心要点
2. 知识回顾：回顾相关的医学知识
3. 逻辑推理：基于知识进行逻辑推理
4. 结论形成：得出最终结论的过程
</thinking>

<answer>
在这里给出最终的专业医学回答
</answer>

请确保推理过程详细、逻辑清晰，让用户能够理解你的思考过程。"""

            if system_prompt:
                cot_system_prompt = system_prompt + "\n\n" + cot_system_prompt

            messages = [
                {"role": "system", "content": cot_system_prompt}
            ]

            # 构建用户消息
            if context:
                user_content = f"参考信息：\n{context}\n\n用户问题：{user_message}"
            else:
                user_content = user_message

            messages.append({"role": "user", "content": user_content})

            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=settings.TEMPERATURE,
                max_tokens=settings.MAX_TOKENS
            )

            full_response = response.choices[0].message.content

            # 解析推理过程和答案
            thinking_process = ""
            final_answer = ""

            if "<thinking>" in full_response and "</thinking>" in full_response:
                thinking_start = full_response.find("<thinking>") + len("<thinking>")
                thinking_end = full_response.find("</thinking>")
                thinking_process = full_response[thinking_start:thinking_end].strip()

            if "<answer>" in full_response and "</answer>" in full_response:
                answer_start = full_response.find("<answer>") + len("<answer>")
                answer_end = full_response.find("</answer>")
                final_answer = full_response[answer_start:answer_end].strip()

            # 如果没有找到标签，尝试其他解析方式
            if not thinking_process and not final_answer:
                # 尝试按段落分割
                lines = full_response.split('\n')
                in_thinking = False
                in_answer = False

                for line in lines:
                    line = line.strip()
                    if '思考' in line or '推理' in line or '分析' in line:
                        in_thinking = True
                        in_answer = False
                        continue
                    elif '回答' in line or '答案' in line or '结论' in line:
                        in_thinking = False
                        in_answer = True
                        continue

                    if in_thinking and line:
                        thinking_process += line + "\n"
                    elif in_answer and line:
                        final_answer += line + "\n"

                # 如果还是没有解析出来，使用原始回复
                if not thinking_process and not final_answer:
                    final_answer = full_response
                    thinking_process = "DeepSeek模型的内部推理过程（模型未明确分离推理和答案部分）"

            return {
                "thinking_process": thinking_process.strip(),
                "final_answer": final_answer.strip(),
                "full_response": full_response,
                "has_explicit_cot": "<thinking>" in full_response
            }

        except Exception as e:
            print(f"CoT推理失败: {e}")
            return {
                "thinking_process": f"推理过程出现错误：{str(e)}",
                "final_answer": f"抱歉，生成回复时出现错误：{str(e)}",
                "full_response": f"错误：{str(e)}",
                "has_explicit_cot": False
            }

    def generate_response_stream(
        self,
        user_message: str,
        context: str = "",
        system_prompt: str = None
    ):
        """
        生成流式回复

        Args:
            user_message: 用户消息
            context: 上下文信息
            system_prompt: 系统提示词

        Yields:
            流式回复的文本片段
        """
        try:
            messages = []

            # 添加系统提示词
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})

            # 构建用户消息
            if context:
                user_content = f"参考信息：\n{context}\n\n用户问题：{user_message}"
            else:
                user_content = user_message

            messages.append({"role": "user", "content": user_content})

            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=settings.TEMPERATURE,
                max_tokens=settings.MAX_TOKENS,
                stream=True  # 启用流式输出
            )

            for chunk in response:
                if chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content

        except Exception as e:
            print(f"流式生成失败: {e}")
            yield f"抱歉，生成回复时出现错误：{str(e)}"

    def test_connection(self) -> bool:
        """测试API连接"""
        try:
            response = self.chat_completion([
                {"role": "user", "content": "你好，请回复'连接成功'"}
            ])
            return "连接成功" in response or "成功" in response
        except Exception as e:
            print(f"连接测试失败: {e}")
            return False


class LLMClient:
    """统一的LLM客户端，支持多种功能"""

    def __init__(self):
        self.deepseek_chat = DeepSeekChatClient()
        self.deepseek_llm = DeepSeekLLM()

    def get_langchain_llm(self):
        """获取LangChain兼容的LLM实例"""
        return self.deepseek_llm

    def generate_response(self, prompt: str, **kwargs) -> str:
        """生成基础回复"""
        return self.deepseek_chat.generate_response(prompt, **kwargs)

    def generate_response_with_cot(self, prompt: str, **kwargs) -> dict:
        """生成包含CoT推理的回复"""
        return self.deepseek_chat.generate_response_with_cot(prompt, **kwargs)

    def generate_response_stream(self, prompt: str, **kwargs):
        """生成流式回复"""
        return self.deepseek_chat.generate_response_stream(prompt, **kwargs)

    def chat_completion(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """聊天完成"""
        return self.deepseek_chat.chat_completion(messages, **kwargs)

    def test_connection(self) -> bool:
        """测试连接"""
        return self.deepseek_chat.test_connection()
