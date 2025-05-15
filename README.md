# AI Agent with Ollama Integration

This project implements an intelligent AI agent system that leverages Ollama models for code analysis, documentation reading, and interactive assistance. The system combines various LLM capabilities with specialized tools for enhanced functionality.

## 🚀 Features

- **Multiple LLM Support**: Integrates with different Ollama models (Mistral and CodeLlama)
- **Document Analysis**: Processes and analyzes PDF documents using LlamaParse
- **Code Reading & Analysis**: Specialized tools for code comprehension
- **Interactive Interface**: Streamlit-based web interface for easy interaction
- **Vector Search**: Efficient document indexing and retrieval using vector store
- **Feedback Analysis**: Built-in feedback management and analysis system

## 📋 Requirements

- Python 3.8+
- Ollama installed and running locally
- LlamaCloud API key for PDF parsing

### Key Dependencies

```txt
streamlit>=1.22.0
llama-index-core==0.10.25
llama-index-llms-ollama==0.1.2
pandas>=2.0.0
torch==2.2.2
transformers==4.39.2
```

For a complete list of dependencies, see `requirements.txt`.

## 🛠️ Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd ai_agent
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
Create a `.env` file in the project root and add:
```
LLAMACLOUD_API_KEY=your_api_key_here
```

## 🚦 Usage

1. Start the Streamlit application:
```bash
streamlit run streamlitApp.py
```

2. The application will be available at `http://localhost:8501`

## 🏗️ Project Structure

- `main.py`: Core initialization and LLM setup
- `streamlitApp.py`: Web interface implementation
- `feedback_manager.py`: Feedback collection and management
- `feedback_analyzer.py`: Analysis of collected feedback
- `model_evaluator.py`: Model performance evaluation
- `code_reader.py`: Code analysis utilities
- `prompts.py`: System prompts and templates
- `model_registry.py`: Model registration and management

## 🔧 Configuration

The system supports multiple LLM models through Ollama:
- Default chat model: Mistral
- Code analysis model: CodeLlama

Models can be configured in the initialization parameters or through the web interface.

## 📊 Features in Detail

### Document Analysis
- Supports PDF document processing
- Vector-based document indexing
- Semantic search capabilities

### Code Analysis
- Code comprehension and explanation
- Documentation generation
- Code quality assessment

### Feedback System
- User feedback collection
- Performance metrics tracking
- Continuous improvement analysis

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests.

## 📝 License

[Add your license information here]

## 🔗 Additional Resources

- [Ollama Documentation](https://ollama.ai/docs)
- [LlamaIndex Documentation](https://docs.llamaindex.ai/)
- [Streamlit Documentation](https://docs.streamlit.io/) 