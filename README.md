# Report Analyst

A modern document analysis tool built with LangChain for analyzing corporate reports and documents.

## Features

- 📄 PDF and document processing
- 🤖 Advanced document analysis using LLMs
- 🔍 Customizable question & answer system
- 📊 Structured report generation
- 🎯 Modular prompt system
- 🚀 FastAPI backend
- ⚡ High performance document processing

## Project Structure

```
report-analyst/
├── app/                    # Main application code
│   ├── api/               # FastAPI routes and endpoints
│   ├── core/              # Core business logic
│   ├── models/            # Pydantic models
│   └── services/          # Service layer
├── prompts/               # Modular prompt templates
│   ├── analysis/         # Document analysis prompts
│   └── qa/               # Q&A prompts
├── config/                # Configuration files
├── data/                  # Data directory
│   ├── input/            # Input documents
│   └── output/           # Generated outputs
└── tests/                # Test suite
```

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file:
```
OPENAI_API_KEY=your_api_key_here
```

4. Run the application:
```bash
uvicorn app.main:app --reload
```

## Usage

1. Place your documents in the `data/input` directory
2. Use the API endpoints to:
   - Analyze documents
   - Ask questions about documents
   - Generate structured reports

## Customizing Prompts

The `prompts` directory contains modular prompt templates that can be customized for different use cases. Each prompt is a separate file that can be modified without affecting the core functionality.

## License

MIT License 