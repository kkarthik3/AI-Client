# 🤖 Universal AI Model Client 

[![Python Version](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](https://opensource.org/licenses/MIT)

A unified Python client for interacting with multiple AI model providers through a single, consistent interface. Currently supports Groq, OpenAI, Together AI, OctoAI, and AWS Bedrock.

## ✨ Features

- 🎯 Single interface for multiple AI providers
- 🔄 Automatic model mapping across providers
- 💾 Built-in caching with TTL
- ⚡ Support for latest LLM models including:
  - LLaMA 3 & 3.1 (various sizes)
  - GPT-4o models
  - Gemma 2
  - Mixtral-7B

## 🛠️ Installation

```bash
pip install cachetools python-dotenv boto3 groq openai together octoai langchain-groq
```

## 🔑 Configuration

Create a `.env` file with your API keys:

```env
GROQ_API_KEY=your_groq_key
OPENAI_API_KEY=your_openai_key
TOGETHER_API_KEY=your_together_key
OCTA_API_KEY=your_octa_key
AWS_ACCESS_KEY_ID=your_aws_access_key
AWS_SECRET_ACCESS_KEY=your_aws_secret_key
```

For AWS Bedrock, provide credentials during client initialization.

## 📖 Usage

```python
# Initialize a client
client = APIClient(
    client_type='groq',  # or 'openai', 'together', 'octa', 'aws'
    groq_api_key='your_key'  # or use environment variables
)

# Create chat messages
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"}
]

# Chat with a specific model
response = client.chat(messages, model='llama3.1-70b')
```

## 🎯 Supported Models

| Model Name | Groq | OpenAI | Together | OctoAI | AWS |
|------------|------|---------|-----------|---------|-----|
| LLaMA 3.1 70B | ✅ | ❌ | ✅ | ✅ | ✅ |
| LLaMA 3.1 8B | ✅ | ❌ | ✅ | ✅ | ✅ |
| LLaMA 3 70B | ✅ | ❌ | ✅ | ✅ | ✅ |
| LLaMA 3 8B | ✅ | ❌ | ✅ | ✅ | ✅ |
| LLaMA 3.1 405B | ✅ | ❌ | ✅ | ✅ | ✅ |
| GPT-4o | ❌ | ✅ | ❌ | ❌ | ❌ |
| Gemma 2 9B | ✅ | ❌ | ❌ | ❌ | ❌ |
| Mixtral 7B | ✅ | ❌ | ❌ | ❌ | ✅ |

## 🚀 Features

- **Unified Interface**: Single API for multiple providers
- **Automatic Model Mapping**: Handles model name translations
- **Built-in Caching**: Uses TTLCache for improved performance
- **Error Handling**: Comprehensive error catching and reporting
- **AWS Integration**: Support for AWS Bedrock with custom credentials

## ⚠️ Error Handling

The client includes comprehensive error handling for:
- Missing API keys
- Unsupported model/provider combinations
- API connection issues
- Invalid message formats

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the [MIT License](LICENSE).