"""
Central place to declare which Ollama models the app supports.
Change the lists here—no need to touch Streamlit code.
"""

# LLMs used for reasoning / general chat
CHAT_MODELS = [
    "mistral",       # default
    "deepseek-r1"
]

# LLMs used for code generation
CODE_MODELS = [
    "codellama",     # default
    "deepseek-coder",
]
