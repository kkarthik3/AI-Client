from octoai.text_gen import ChatMessage
from octoai.client import OctoAI
from together import Together
from os import environ as env
from openai import OpenAI
from cachetools import cached, TTLCache
from dotenv import load_dotenv
from groq import Groq
from langchain_aws import ChatBedrock
from langchain.schema import (
    SystemMessage,
    HumanMessage,
    AIMessage
)

load_dotenv()
cache = TTLCache(maxsize=100, ttl=1800)

class APIClient:
    def __init__(self, client_type,AWS_REGION=None,AWS_ACCESS_KEY_ID=None,AWS_SECRET_ACCESS_KEY=None,other_model_api_keys = None):
        
        """
        Initialize an instance of the APIClient class.

        Args:
            client_type (str): The type of the client to be initialized. Supported values are "groq", "openai", "octa", "together", and "aws".
            AWS_REGION (str, optional): The AWS region to use. Defaults to the value of the AWS_REGION environment variable.
            AWS_ACCESS_KEY_ID (str, optional): The AWS access key ID to use. Defaults to the value of the AWS_ACCESS_KEY_ID environment variable.
            AWS_SECRET_ACCESS_KEY (str, optional): The AWS secret access key to use. Defaults to the value of the AWS_SECRET_ACCESS_KEY environment variable.
            other_model_api_keys (str, optional): The API key to use for models other than the default Groq model. Defaults to the value of the environment variable with the name specified in the client_env_map dictionary.

        Raises:
            ValueError: If the client_type is not recognized or if the required credentials are missing.

        """
        
        self.region = AWS_REGION or env.get('AWS_REGION')
        self.access = AWS_ACCESS_KEY_ID or env.get('AWS_ACCESS_KEY_ID')
        self.secret = AWS_SECRET_ACCESS_KEY or env.get('AWS_SECRET_ACCESS_KEY')
        self.client_type = client_type.lower()
        

        self.client_env_map = {
            "groq": "GROQ_API_KEY",
            "openai": "OPENAI_API_KEY",
            "octa": "OCTA",
            "together": "TOGETHER"
        }

        if self.client_type in self.client_env_map:
            self.other_model_api_keys = other_model_api_keys or env.get(self.client_env_map[self.client_type])
            if not self.other_model_api_keys:
                raise ValueError(
                    f"Though you chose {self.client_type.upper()}, you must provide the other_model_api_keys "
                    f"or set it as {self.client_env_map[self.client_type]} in environment variables"
                )
            
        if self.client_type == "aws":
            # Check if any required AWS credentials are missing
            missing_keys = [key for key, value in {
                'AWS_REGION': self.region,
                'AWS_ACCESS_KEY_ID': self.access,
                'AWS_SECRET_ACCESS_KEY': self.secret
            }.items() if not value]

            if missing_keys:
                raise ValueError(
                    f"Though you chose AWS, you must provide the following missing credentials: {', '.join(missing_keys)} "
                    f"either as input or environment variables."
                )
        
    
        self.client = self._initialize_client()

        self.model_mapping = {
                "llama3.1-70b": {
                    "groq": "llama-3.1-70b-versatile",
                    "openai": None,
                    "together": "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
                    "octa": "meta-llama-3.1-70b-instruct",
                    "aws": "meta.llama3-1-70b-instruct-v1:0"
                },
                "llama3.1-8b": {
                    "groq": "llama-3.1-8b-instant",
                    "openai": None,
                    "together": "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
                    "octa": "meta-llama-3.1-8b-instruct",
                    "aws": "meta.llama3-1-8b-instruct-v1:0"
                },
                "llama3-70b": {
                    "groq": "llama3-70b-8192",
                    "openai": None,
                    "together": "meta-llama/Meta-Llama-3-70B-Instruct-Turbo",
                    "octa": "meta-llama-3-70b-instruct",
                    "aws": "meta.llama3-70b-instruct-v1:0"
                },
                "llama3-8b": {
                    "groq": "llama3-8b-8192",
                    "openai": None,
                    "together": "meta-llama/Meta-Llama-3-8B-Instruct-Turbo",
                    "octa": "meta-llama-3-8b-instruct",
                    "aws": "meta.llama3-8b-instruct-v1:0"
                },
                "llama3.1-405b": {
                    "groq": "llama-3.1-405b-reasoning",
                    "openai": None,
                    "together": "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
                    "octa": "meta-llama-3.1-405b-instruct",
                    "aws": "meta.llama3-1-405b-instruct-v1:0"
                },
                "gpt-4o": {
                    "groq": None,
                    "openai": "gpt-4o",
                    "together": None,
                    "octa": None,
                    "aws": None
                },
                "gpt-4o-mini": {
                    "groq": None,
                    "openai": "gpt-4o-mini",
                    "together": None,
                    "octa": None,
                    "aws": None
                },
                "gemma2-9b": {
                    "groq": "gemma2-9b-it",
                    "openai": None,
                    "together": None,
                    "octa": None,
                    "aws": None
                },
                "mixtral-7b": {
                    "groq": "mixtral-8x7b-32768",
                    "openai": None,
                    "together": None,
                    "octa": None,
                    "aws": "mistral.mixtral-8x7b-instruct-v0:1"
                }
            }
    
    
    @cached(cache)
    def _initialize_client(self):
        if self.client_type.lower() == 'groq':
            return self._groq_client()
        elif self.client_type.lower() == 'openai':
            return self._openai_client()
        elif self.client_type.lower() == 'together':
            return self._together_ai()
        elif self.client_type.lower() == 'octa':
            return self._octa()
        elif self.client_type.lower() == 'aws':
            pass
        else:
            raise ValueError("Unsupported client type. Please use 'groq', 'openai', 'together' or 'octa' or 'AWS'.")

    @cached(cache)
    def _groq_client(self):
        return Groq(api_key=self.other_model_api_keys)

    @cached(cache)
    def _openai_client(self):
        return OpenAI(api_key=self.other_model_api_keys)
    
    @cached(cache)
    def _together_ai(self):
        return Together(api_key=self.other_model_api_keys)
    
    @cached(cache)
    def _octa(self):
        octaclient = OctoAI(
        api_key=env.get('OCTA'))  
        return octaclient
    
    @cached(cache)
    def _aws_bedrock_client(self, model):
        return ChatBedrock(
            model_id=model,
            model_kwargs=dict(temperature=0),
            region_name=self.region
        )
    
    def chat(self, messages, model):
        specific_model = self.model_mapping.get(model, {}).get(self.client_type)
        if not specific_model:
            raise ValueError(f"Model {model} not supported for client {self.client_type}")


        if self.client_type in ['groq', 'openai', 'together']:
            # Assuming Groq API has similar method structure
            return self.client.chat.completions.create(
                messages=messages,
                model=specific_model,
                temperature=0,
                stream=False
            )

        elif self.client_type =='octa':
            return self.client.text_gen.create_chat_completion(
            model=specific_model,
            messages=[ChatMessage(role=data["role"],content=data["content"])
                        for data in messages]
            )
        elif self.client_type =='aws':
            llm = self._aws_bedrock_client(specific_model)
    
            role_to_message = {
                "system": SystemMessage,
                "user": HumanMessage,
                "assistant": AIMessage
            }

            messages = [role_to_message[db_message["role"]](content=str(db_message["content"])) for db_message in messages if db_message["role"] in role_to_message]
            messages
            return llm.invoke(messages)
        else:
            raise ValueError("Unsupported client type. Please use 'groq', 'openai', 'together' or 'octa'.")