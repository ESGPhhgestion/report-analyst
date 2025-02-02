from pathlib import Path
from typing import List, Dict, Optional, AsyncGenerator, Any
import os
from dotenv import load_dotenv
import shutil
import yaml
import logging
import sys
import json

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
import chromadb

from .prompt_manager import PromptManager

# Setup logging at the top of the file
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Check for required environment variables
if not os.getenv("OPENAI_API_KEY"):
    logger.error("OPENAI_API_KEY environment variable is not set")
    raise ValueError("OPENAI_API_KEY environment variable is required")

if not os.getenv("OPENAI_ORGANIZATION"):
    logger.error("OPENAI_ORGANIZATION environment variable is not set")
    raise ValueError("OPENAI_ORGANIZATION environment variable is required")

def log_analysis_step(message: str, level: str = "info"):
    """Helper function to log analysis steps with consistent formatting"""
    log_func = getattr(logger, level)
    log_func(f"[ANALYSIS] {message}")

class DocumentAnalyzer:
    def __init__(self):
        self.prompt_manager = PromptManager()
        
        model_name = os.getenv("OPENAI_API_MODEL", "gpt-4-turbo-preview")
        log_analysis_step(f"Using model: {model_name}")
        
        try:
            self.llm = ChatOpenAI(
                temperature=0,
                model=model_name,
                api_key=os.getenv("OPENAI_API_KEY"),
                organization=os.getenv("OPENAI_ORGANIZATION")
            )
            
            self.embeddings = OpenAIEmbeddings(
                api_key=os.getenv("OPENAI_API_KEY"),
                organization=os.getenv("OPENAI_ORGANIZATION")
            )
            
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=20
            )
            
        except Exception as e:
            log_analysis_step(f"Error initializing OpenAI clients: {str(e)}", "error")
            raise
        
        self.questions = self._load_questions()

    def _load_questions(self) -> dict:
        """Load TCFD questions from YAML files"""
        # Look in app/questionsets first, then try questionsets
        possible_paths = [
            Path(__file__).parent.parent / "questionsets" / "tcfd_questions.yaml",  # app/questionsets
            Path("questionsets") / "tcfd_questions.yaml"  # questionsets in root
        ]
        
        yaml_file = None
        for path in possible_paths:
            if path.exists():
                yaml_file = path
                break
                
        if not yaml_file:
            log_analysis_step(f"Could not find questions file in any of: {[str(p) for p in possible_paths]}", "error")
            return {}
            
        log_analysis_step(f"Loading questions from {yaml_file}", "debug")
        
        with open(yaml_file, 'r') as f:
            config = yaml.safe_load(f)
            questions = {}
            
            # Convert the questions list into a structured format
            for q in config.get('questions', []):
                q_id = q.get('id', '')
                if q_id:
                    questions[q_id] = {
                        'text': q.get('text', ''),
                        'guidelines': q.get('guidelines', '')
                    }
            
            log_analysis_step(f"Loaded {len(questions)} questions", "debug")
            return questions

    async def process_document(self, file_path: str, question_ids: List[int] = None) -> AsyncGenerator[Dict, None]:
        """Process document and analyze TCFD questions"""
        log_analysis_step(f"Starting document processing: {file_path}")
        log_analysis_step(f"Processing questions: {question_ids}")
        
        try:
            # Initial status
            yield {"status": "Starting analysis..."}
            
            log_analysis_step("Loading PDF document")
            yield {"status": "Loading PDF document..."}
            
            loader = PyPDFLoader(str(file_path))
            pages = loader.load()
            log_analysis_step(f"Loaded {len(pages)} pages")
            yield {"status": f"✓ Loaded {len(pages)} pages"}
            
            log_analysis_step("Splitting text into chunks")
            yield {"status": "Splitting text into chunks..."}
            texts = self.text_splitter.split_documents(pages)
            log_analysis_step(f"Created {len(texts)} text chunks")
            yield {"status": f"✓ Created {len(texts)} text chunks"}
            
            log_analysis_step("Creating vector store")
            yield {"status": "Creating vector store..."}
            vectorstore = Chroma.from_documents(texts, self.embeddings)
            log_analysis_step("Vector store created successfully")
            yield {"status": "✓ Vector store created successfully"}
            
            # Process each question
            for q_id in question_ids:
                question_key = f"tcfd_{q_id}"
                if question_key not in self.questions:
                    continue
                    
                question_data = self.questions[question_key]
                log_analysis_step(f"Processing question {q_id}")
                yield {"status": f"Analyzing question {q_id}"}
                
                # Get relevant context using TOP_K=20
                docs = vectorstore.similarity_search(question_data['text'], k=20)
                context = "\n".join(d.page_content for d in docs)
                log_analysis_step(f"Retrieved {len(docs)} relevant chunks for question {q_id}", "debug")
                
                # Get LLM response
                messages = self.prompt_manager.get_analysis_messages(
                    question=question_data['text'],
                    context=context,
                    guidelines=question_data['guidelines']
                )
                result = await self.llm.ainvoke(messages)
                log_analysis_step(f"Got LLM response for question {q_id}", "debug")
                
                # Extract JSON from response
                try:
                    result_text = result.content
                    json_start = result_text.rfind('{')
                    json_end = result_text.rfind('}') + 1
                    if json_start >= 0 and json_end > json_start:
                        result_text = result_text[json_start:json_end]
                    
                    result_json = json.loads(result_text)
                    
                    # Ensure we have all required keys
                    required_keys = ["ANSWER", "SCORE", "EVIDENCE", "GAPS", "SOURCES"]
                    missing_keys = [key for key in required_keys if key not in result_json]
                    if missing_keys:
                        raise ValueError(f"Missing required keys in response: {missing_keys}")
                    
                    # Return the result in the exact format expected by display code
                    yield {
                        "question_number": q_id,
                        "result": json.dumps({
                            "ANSWER": result_json["ANSWER"],
                            "SCORE": result_json["SCORE"],
                            "EVIDENCE": result_json["EVIDENCE"],
                            "GAPS": result_json["GAPS"],
                            "SOURCES": result_json["SOURCES"]
                        })
                    }
                    
                except Exception as e:
                    log_analysis_step(f"Error processing result for question {q_id}: {str(e)}", "error")
                    yield {
                        "question_number": q_id,
                        "result": json.dumps({
                            "ANSWER": "Error processing analysis response",
                            "SCORE": 0,
                            "EVIDENCE": [],
                            "GAPS": ["Error processing response"],
                            "SOURCES": []
                        })
                    }
                    
        except Exception as e:
            log_analysis_step(f"Error processing document: {str(e)}", "error")
            yield {"error": f"Failed to process document: {str(e)}"} 