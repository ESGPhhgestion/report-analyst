from pathlib import Path
from typing import Dict, Optional
import yaml
from langchain.prompts import PromptTemplate
from ..models.requests import AnalysisType

class PromptManager:
    def __init__(self, prompts_dir: str = "prompts"):
        self.prompts_dir = Path(prompts_dir)
        self.analysis_prompts: Dict[str, PromptTemplate] = {}
        self.qa_prompt: Optional[PromptTemplate] = None
        self._load_prompts()

    def _load_prompts(self):
        """Load all prompts from the prompts directory"""
        # Load analysis prompts
        analysis_dir = self.prompts_dir / "analysis"
        if analysis_dir.exists():
            for prompt_file in analysis_dir.glob("*.yaml"):
                with open(prompt_file) as f:
                    prompt_data = yaml.safe_load(f)
                    self.analysis_prompts[prompt_file.stem] = PromptTemplate(
                        template=prompt_data["template"],
                        input_variables=prompt_data["input_variables"]
                    )

        # Load QA prompt
        qa_prompt_path = self.prompts_dir / "qa" / "default.yaml"
        if qa_prompt_path.exists():
            with open(qa_prompt_path) as f:
                prompt_data = yaml.safe_load(f)
                self.qa_prompt = PromptTemplate(
                    template=prompt_data["template"],
                    input_variables=prompt_data["input_variables"]
                )
        else:
            # Fallback to default prompts if files don't exist
            self._setup_default_prompts()

    def _setup_default_prompts(self):
        """Set up default prompts if no files are found"""
        # Default analysis prompt
        default_analysis_template = """You are a professional document analyzer.
        Please analyze the following document with focus on {analysis_type} aspects.
        
        Key points to consider:
        - Main themes and topics
        - Key findings and insights
        - Important data points and statistics
        - Recommendations and conclusions
        
        Document: {context}
        
        Provide a detailed analysis focusing on these aspects."""

        self.analysis_prompts["default"] = PromptTemplate(
            template=default_analysis_template,
            input_variables=["analysis_type", "context"]
        )

        # Default QA prompt
        default_qa_template = """You are a helpful assistant answering questions about documents.
        Use the following pieces of context to answer the question at the end.
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        
        Context: {context}
        
        Question: {question}
        
        Answer:"""

        self.qa_prompt = PromptTemplate(
            template=default_qa_template,
            input_variables=["context", "question"]
        )

    def get_analysis_prompt(self, analysis_type: AnalysisType) -> PromptTemplate:
        """Get the appropriate analysis prompt for the given type"""
        prompt_key = analysis_type.value
        return self.analysis_prompts.get(prompt_key, self.analysis_prompts["default"])

    def get_qa_prompt(self) -> PromptTemplate:
        """Get the QA prompt"""
        return self.qa_prompt

    def add_custom_prompt(self, name: str, template: str, input_variables: list):
        """Add a custom prompt template"""
        self.analysis_prompts[name] = PromptTemplate(
            template=template,
            input_variables=input_variables
        )

    def get_analysis_messages(self, question: str, context: str, guidelines: str, analysis_guidelines: str) -> list:
        """Generate messages for TCFD question analysis"""
        return [
            {
                "role": "system",
                "content": f"""You are a sustainability report analyst. Analyze the provided context to answer TCFD questions.

Analysis Guidelines:
{analysis_guidelines}

Question-Specific Guidelines:
{guidelines}"""
            },
            {
                "role": "user",
                "content": f"""Question: {question}

Context from report:
{context}

Provide your analysis in JSON format:
{{
    "answer": "Your detailed analysis",
    "score": "Score from 0-10 indicating disclosure quality",
    "evidence": ["List of specific evidence from the text"],
    "gaps": ["List of missing information or gaps in disclosure"]
}}"""
            }
        ] 