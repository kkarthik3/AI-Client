from octoai.text_gen import ChatMessage
from octoai.client import OctoAI
from together import Together
from os import environ as env
from openai import OpenAI
from cachetools import cached, TTLCache
from functools import lru_cache
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from groq import Groq
import boto3

load_dotenv()
cache = TTLCache(maxsize=100, ttl=1800)

class APIClient:
    def __init__(self, client_type, region=None, access=None, secret=None, groq_api_key=None, openai_api_key=None, together_api_key=None, octa_api_key=None):
        self.region = region
        self.access = access
        self.secret = secret
        self.client_type = client_type.lower()
        self.groq_api_key = groq_api_key or env.get("GROQ_API_KEY")
        self.openai_api_key = openai_api_key or env.get("OPENAI_API_KEY")
        self.together_api_key = together_api_key or env.get("Together")
        self.octa_api_key = octa_api_key or env.get("octa")
        self.client = self.initialize_client()
        self.model_mapping = {
            'llama3.1-70b': {
                'groq': 'llama-3.1-70b-versatile',
                'openai': None,
                'together': "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
                'octa': 'meta-llama-3.1-70b-instruct',
                'aws': 'meta.llama3-1-70b-instruct-v1:0'
            },
            'llama3.1-8b': {
                'groq': 'llama-3.1-8b-instant',
                'openai': None,
                'together': "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
                'octa': 'meta-llama-3.1-8b-instruct',
                'aws': 'meta.llama3-1-8b-instruct-v1:0'
            },
            'llama3-70b': {
                'groq': 'llama3-70b-8192',
                'openai': None,
                'together': "meta-llama/Meta-Llama-3-70B-Instruct-Turbo",
                'octa': 'meta-llama-3-70b-instruct',
                'aws': 'meta.llama3-70b-instruct-v1:0'
            },
            'llama3-8b': {
                'groq': 'llama3-8b-8192',
                'openai': None,
                'together': "meta-llama/Meta-Llama-3-8B-Instruct-Turbo",
                'octa': 'meta-llama-3-8b-instruct',
                'aws': 'meta.llama3-8b-instruct-v1:0'
            },
            'llama3.1-405b': {
                'groq': 'llama-3.1-405b-reasoning',
                'openai': None,
                'together': "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
                'octa': 'meta-llama-3.1-405b-instruct',
                'aws': 'meta.llama3-1-405b-instruct-v1:0'
            },
            'gpt-4o': {
                'groq': None,
                'openai': "gpt-4o",
                'together': None,
                'octa': None,
                'aws': None
            },
            'gpt-4o-mini': {
                'groq': None,
                'openai': "gpt-4o-mini",
                'together': None,
                'octa': None,
                'aws': None
            },
            'gemma2-9b': {
                'groq': "gemma2-9b-it",
                'openai': None,
                'together': None,
                'octa': None,
                'aws': None
            },
            'mixtral-7b': {
                'groq': "mixtral-8x7b-32768",
                'openai': None,
                'together': None,
                'octa': None,
                'aws': 'mistral.mixtral-8x7b-instruct-v0:1'
            }
            # Add other models and their respective mappings here
        }

    @cached(cache)
    def initialize_client(self):
        try:
            if self.client_type == 'groq':
                return self.groq_client()
            elif self.client_type == 'openai':
                return self.openai_client()
            elif self.client_type == 'together':
                return self.together_ai()
            elif self.client_type == 'octa':
                return self.octa()
            elif self.client_type == "aws":
                return self.aws(self.region, self.access, self.secret)
            else:
                raise ValueError("Unsupported client type. Please use 'groq', 'openai', 'together', 'octa', or 'aws'.")
        except Exception as e:
            print(f"Error initializing client: {e}")
            raise

    @cached(cache)
    def groq_client(self):
        try:
            if not self.groq_api_key:
                raise ValueError("GROQ API key not set.")
            return Groq(api_key=self.groq_api_key)
        except Exception as e:
            print(f"Error creating Groq client: {e}")
            raise

    @cached(cache)
    def openai_client(self):
        try:
            if not self.openai_api_key:
                raise ValueError("OPENAI API key not set.")
            return OpenAI(api_key=self.openai_api_key)
        except Exception as e:
            print(f"Error creating OpenAI client: {e}")
            raise
    
    @cached(cache)
    def together_ai(self):
        try:
            if not self.together_api_key:
                raise ValueError("TOGETHER API key not set.")
            return Together(api_key=self.together_api_key)
        except Exception as e:
            print(f"Error creating Together AI client: {e}")
            raise
    
    @cached(cache)
    def octa(self):
        try:
            if not self.octa_api_key:
                raise ValueError("OCTA API key not set.")
            return OctoAI(api_key=self.octa_api_key)
        except Exception as e:
            print(f"Error creating Octa client: {e}")
            raise
    
    @cached(cache)
    def aws(self, region, access, secret):  
        try:
            if region and access and secret:
                client = boto3.client(
                    service_name='bedrock-runtime', 
                    region_name=region,
                    aws_access_key_id=access,
                    aws_secret_access_key=secret
                )
                return client
            else:
                raise ValueError("Please provide AWS Region, Access, and Secret Key.")
        except Exception as e:
            print(f"Error creating AWS client: {e}")
            raise
    
    def chat(self, messages, model):
        try:
            specific_model = self.model_mapping.get(model, {}).get(self.client_type)
            if not specific_model:
                raise ValueError(f"Model {model} not supported for client {self.client_type}")

            if self.client_type in ['groq', 'openai', 'together']:
                return self.client.chat.completions.create(
                    messages=messages,
                    model=specific_model,
                    temperature=0,
                    stream=False
                )
            elif self.client_type == 'octa':
                return self.client.text_gen.create_chat_completion(
                    model=specific_model,
                    messages=[ChatMessage(role=data["role"], content=data["content"]) for data in messages]
                )
            elif self.client_type == 'aws':
                modified_message = [
                    {"role": data["role"], "content": [{"text": data["content"]}]}
                    for data in messages if data["role"] != "system"
                ]
                system_messages = [{"text": data["content"]} for data in messages if data["role"] == "system"]
                params = {
                    "modelId": specific_model,
                    "messages": modified_message,
                    "inferenceConfig": {"temperature": 0},
                    "system": system_messages
                }
                return self.client.converse(**params)
            else:
                raise ValueError("Unsupported client type. Please use 'groq', 'openai', 'together', 'octa', or 'aws'.")
        except Exception as e:
            print(f"Error during chat: {e}")
            raise