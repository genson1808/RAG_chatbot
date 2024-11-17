import os
from dotenv import load_dotenv, find_dotenv

from langchain_groq import ChatGroq
from langchain.llms import OpenAI

load_dotenv(find_dotenv())

class SingletonChatLLM:
    _instance = None

    def __init__(self, llm_name:str="CHAT_GROQ"):
        if llm_name == 'CHAT_OPENAI':
            self.llm = OpenAI(
                model="gpt-3.5-turbo",
                temperature=0,
                max_token=500,
                openai_api_key=os.getenv('OPENAI_API_KEY')
            )
        elif llm_name == 'CHAT_GROQ':
            self.llm = ChatGroq(
                model="gemma2-9b-it", # llama3-8b-8192, llama3-70b-8192, llama-3.1-8b-instant, llama-3.1-70b-versatile, gemma2-9b-it, mixtral-8x7b-32768
                temperature=0,
                max_retries=2,
                api_key=os.getenv('GROQ_API_KEY')
            )
        else:
            self.llm = ChatGroq(
                model="gemma2-9b-it", # llama3-8b-8192, llama3-70b-8192, llama-3.1-8b-instant, llama-3.1-70b-versatile, gemma2-9b-it, mixtral-8x7b-32768
                temperature=0,
                max_retries=2,
                api_key=os.getenv('GROQ_API_KEY')
            )


    def __new__(cls, *args, **kwargs):
        if cls._instance == None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def get_llm(self):
        return self.llm