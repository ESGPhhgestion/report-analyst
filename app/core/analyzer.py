from pathlib import Path
from typing import List, Dict, Optional, AsyncGenerator
import os
from dotenv import load_dotenv
import shutil
import yaml
import logging

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
import chromadb

from ..models.responses import AnalysisResponse, QuestionResponse
from ..models.requests import AnalysisType, DocumentMetadata
from .document_processor import DocumentProcessor
from .prompt_manager import PromptManager

# Setup logging at the top of the file
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class DocumentAnalyzer:
    def __init__(self):
        logger.info("Initializing DocumentAnalyzer")
        self.document_processor = DocumentProcessor()
        self.prompt_manager = PromptManager()
        
        model_name = os.getenv("OPENAI_API_MODEL", "gpt-4-turbo-preview")
        logger.info(f"Using OpenAI model: {model_name}")
        
        self.llm = ChatOpenAI(
            temperature=0,
            model_name=model_name,
            api_key=os.getenv("OPENAI_API_KEY"),
            organization=os.getenv("OPENAI_ORGANIZATION")
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=200
        )
        self.embeddings = OpenAIEmbeddings(
            api_key=os.getenv("OPENAI_API_KEY"),
            organization=os.getenv("OPENAI_ORGANIZATION")
        )
        self.questions = self._load_questions()

    def _load_questions(self) -> dict:
        """Load TCFD questions from YAML files"""
        question_dir = Path(__file__).parent.parent / "questionsets"
        logger.info(f"Loading questions from {question_dir}")
        
        yaml_file = question_dir / "tcfd_questions.yaml"
        logger.info(f"Loading questions from file: {yaml_file}")
        
        with open(yaml_file, 'r') as f:
            config = yaml.safe_load(f)
            logger.info(f"Loaded {len(config['questions'])} questions")
            
            # Convert the flat list into a structured format
            questions = {
                "TCFD Analysis": {
                    "questions": [q['text'] for q in config['questions']],
                    "guidelines": [q['guidelines'] for q in config['questions']],
                    "analysis_guidelines": config.get('analysis_guidelines', '')
                }
            }
            
            logger.info("Questions loaded successfully")
            return questions

    async def process_document(self, file_path: str, question_ids: List[int] = None) -> AsyncGenerator[Dict, None]:
        """Process document and analyze TCFD questions"""
        logger.info(f"Starting document processing for: {file_path} with question IDs: {question_ids}")
        try:
            # Initial status
            logger.info("Starting analysis...")
            yield {"status": "Starting analysis..."}
            
            logger.info("Loading PDF document...")
            yield {"status": "Loading PDF document..."}
            
            loader = PyPDFLoader(str(file_path))
            pages = loader.load()
            logger.info(f"Loaded {len(pages)} pages")
            yield {"status": f"✓ Loaded {len(pages)} pages"}
            
            logger.info("Splitting text into chunks...")
            yield {"status": "Splitting text into chunks..."}
            texts = self.text_splitter.split_documents(pages)
            logger.info(f"Created {len(texts)} text chunks")
            yield {"status": f"✓ Created {len(texts)} text chunks"}
            
            logger.info("Creating vector store...")
            yield {"status": "Creating vector store..."}
            vectorstore = Chroma.from_documents(texts, self.embeddings)
            logger.info("Vector store created successfully")
            yield {"status": "✓ Vector store created successfully"}
            
            # Process each section and question
            for section, config in self.questions.items():
                section_name = section
                logger.info(f"Starting section: {section_name}")
                
                # Get all questions and guidelines
                all_questions = config['questions']
                all_guidelines = config['guidelines']
                analysis_guidelines = config.get('analysis_guidelines', '')
                
                # Filter questions and guidelines based on selected IDs
                if question_ids is not None:
                    logger.info(f"Filtering for selected question IDs: {question_ids}")
                    selected_pairs = [
                        (q, g) for i, (q, g) in enumerate(zip(all_questions, all_guidelines), 1)
                        if i in question_ids
                    ]
                    if not selected_pairs:
                        logger.warning("No questions selected after filtering")
                        continue
                    questions_list, guidelines_list = zip(*selected_pairs)
                    logger.info(f"Selected {len(questions_list)} questions")
                else:
                    questions_list = all_questions
                    guidelines_list = all_guidelines
                    logger.info("Using all questions (no filtering)")
                
                total_questions = len(questions_list)
                
                # Process selected questions
                for i, (question, guidelines) in enumerate(zip(questions_list, guidelines_list), 1):
                    question_num = question_ids[i-1] if question_ids else i
                    logger.info(f"Processing question {question_num}")
                    yield {"status": f"Analyzing {section_name} question {question_num}"}
                    
                    # Get relevant context
                    docs = vectorstore.similarity_search(question, k=3)
                    context = "\n".join(d.page_content for d in docs)
                    
                    # Get LLM response
                    messages = self.prompt_manager.get_analysis_messages(
                        question=question,
                        context=context,
                        guidelines=guidelines,
                        analysis_guidelines=analysis_guidelines
                    )
                    result = await self.llm.ainvoke(messages)
                    
                    result_dict = {
                        "section": section_name,
                        "question": question,
                        "question_number": question_num,
                        "total_questions": len(question_ids) if question_ids else total_questions,
                        "result": result.content
                    }
                    logger.info(f"Completed question {question_num}")
                    yield result_dict
                    
        except Exception as e:
            logger.error(f"Error processing document: {str(e)}", exc_info=True)
            yield {"error": f"Failed to process document: {str(e)}"}

    async def analyze(
        self,
        document_id: str,
        analysis_type: AnalysisType
    ) -> AnalysisResponse:
        """Analyze a document using the specified analysis type"""
        try:
            document_path = self.document_processor.get_document_path(document_id)
            logger.info(f"Found document path: {document_path}")
            
            logger.debug("Loading and splitting document...")
            loader = PyPDFLoader(document_path)
            pages = loader.load_and_split()
            texts = [str(doc.page_content) for doc in pages]
            
            logger.debug(f"Document split into {len(texts)} chunks")
            logger.debug(f"First chunk: {texts[0][:50]}...")
            logger.debug(f"Last chunk: {texts[-1][:50]}...")
            
            # Clear any existing vector store
            output_dir = f"data/output/{document_id}"
            if os.path.exists(output_dir):
                logger.debug(f"Cleared existing vector store at {output_dir}")
                shutil.rmtree(output_dir)
            
            collection_name = f"{document_id}_{analysis_type}"
            logger.debug(f"Creating vector store with collection name: {collection_name}")
            
            vectorstore = Chroma.from_documents(
                documents=texts,
                collection_name=collection_name
            )

            # Get analysis prompt
            analysis_prompt = self.prompt_manager.get_analysis_prompt(analysis_type)
            
            # Create analysis chain
            chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=vectorstore.as_retriever(),
                chain_type_kwargs={
                    "prompt": analysis_prompt
                }
            )

            # Run analysis
            result = chain.run("Analyze this document")

            # Get summary
            summary_chain = load_summarize_chain(
                llm=self.llm,
                chain_type="map_reduce"
            )
            summary = summary_chain.run(texts)

            # Extract key points and topics
            key_points = self._extract_key_points(result)
            topics = self._extract_topics(result)

            return AnalysisResponse(
                document_id=document_id,
                analysis_type=analysis_type.value,
                summary=summary,
                key_points=key_points,
                topics=topics,
                metadata=await self._get_metadata(document_path),
                confidence_score=0.85  # This could be calculated based on various factors
            )
        except Exception as e:
            logger.error(f"Error analyzing document: {e}")
            raise

    async def ask_question(
        self,
        document_id: str,
        question: str,
        context: Optional[str] = None
    ) -> QuestionResponse:
        """Ask a question about a document"""
        doc_path = await self.document_processor.get_document_path(document_id)
        if not doc_path:
            raise ValueError(f"Document {document_id} not found")

        # Load and split the document
        loader = PyPDFLoader(str(doc_path))
        pages = loader.load()
        texts = self.text_splitter.split_documents(pages)

        # Create vector store
        vectorstore = Chroma.from_documents(
            documents=texts,
            collection_name=f"{document_id}_qa"
        )

        # Get QA prompt
        qa_prompt = self.prompt_manager.get_qa_prompt()
        
        # Create QA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(),
            chain_type_kwargs={
                "prompt": qa_prompt
            }
        )

        # Run question
        result = qa_chain.run(question)

        return QuestionResponse(
            document_id=document_id,
            question=question,
            answer=result,
            context_used=context,
            confidence_score=0.9,  # This could be calculated based on various factors
            relevant_quotes=self._extract_quotes(result)
        )

    def _extract_key_points(self, analysis_result: str) -> List[str]:
        """Extract key points from the analysis result"""
        # This is a simple implementation - could be more sophisticated
        points = []
        for line in analysis_result.split("\n"):
            if line.strip().startswith("-"):
                points.append(line.strip()[2:])
        return points[:5]  # Return top 5 key points

    def _extract_topics(self, analysis_result: str) -> List[Dict[str, float]]:
        """Extract topics and their relevance scores"""
        # This is a simple implementation - could be more sophisticated
        topics = []
        # Mock topics for now - in reality, this would use topic modeling
        topics = [
            {"sustainability": 0.8},
            {"finance": 0.6},
            {"innovation": 0.4}
        ]
        return topics

    def _extract_quotes(self, result: str) -> List[str]:
        """Extract relevant quotes from the result"""
        quotes = []
        current_quote = ""
        in_quote = False
        
        for line in result.split("\n"):
            if '"' in line:
                quotes.append(line.strip())
        
        return quotes[:3]  # Return top 3 quotes

    async def _get_metadata(self, document_path: str) -> dict:
        try:
            metadata = await self.document_processor.extract_metadata(document_path)
            if not metadata:
                return {
                    "file_type": "unknown",
                    "file_size": 0,
                    "title": None,
                    "author": None,
                    "date": None,
                    "num_pages": 0
                }
            return metadata
        except Exception as e:
            logger.error(f"Error extracting metadata: {e}")
            return {
                "file_type": "unknown",
                "file_size": 0,
                "title": None,
                "author": None,
                "date": None,
                "num_pages": 0
            } 