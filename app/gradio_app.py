import gradio as gr
from pathlib import Path
import os
from dotenv import load_dotenv
import tempfile
import shutil
import uuid
import json
from typing import AsyncGenerator, Dict, List
import logging

from .core.analyzer import DocumentAnalyzer

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Log to console
        logging.FileHandler('app.log')  # Log to file
    ]
)
logger = logging.getLogger(__name__)

class DocumentService:
    def __init__(self):
        self.analyzer = DocumentAnalyzer()
        # Get valid question IDs from the loaded questions
        self.valid_question_ids = list(range(1, len(self.analyzer.questions["TCFD Analysis"]["questions"]) + 1))
        logger.info(f"Initialized with {len(self.valid_question_ids)} valid question IDs")

    def validate_question_ids(self, question_ids: List[int]) -> List[int]:
        """Validate and filter question IDs"""
        if not question_ids:
            raise ValueError("No questions selected")
            
        valid_ids = [qid for qid in question_ids if qid in self.valid_question_ids]
        if not valid_ids:
            raise ValueError("No valid questions selected")
            
        logger.info(f"Validated question IDs: {valid_ids}")
        return valid_ids

    async def process_document(self, file_path: str, question_ids: List[int] = None) -> AsyncGenerator[Dict, None]:
        """Process uploaded document and stream analysis results"""
        if not file_path:
            yield {"error": "No file uploaded"}
            return
            
        try:
            # Validate question IDs only if they are provided
            if question_ids is not None:
                question_ids = self.validate_question_ids(question_ids)
                logger.info(f"Processing questions: {question_ids}")
            else:
                # If no questions specified, use all valid IDs
                question_ids = self.valid_question_ids
                logger.info("No questions specified, using all questions")
            
            temp_file = Path(tempfile.gettempdir()) / f"temp_{uuid.uuid4()}.pdf"
            try:
                shutil.copy2(file_path, temp_file)
                async for result in self.analyzer.process_document(str(temp_file), question_ids):
                    logger.info(f"Processing section: {result.get('section', 'unknown')}")
                    yield result
            finally:
                if temp_file.exists():
                    temp_file.unlink()
        except Exception as e:
            logger.error(f"Error processing document: {str(e)}")
            yield {"error": f"Failed to process document: {str(e)}"}

def create_app():
    service = DocumentService()
    progress_tracker = gr.Progress()
    
    with gr.Blocks(title="TCFD Report Analyzer", theme=gr.themes.Soft()) as app:
        gr.Markdown("# TCFD Report Analyzer")
        
        with gr.Tabs() as tabs:
            # Analysis Tab
            with gr.Tab("Analysis"):
                gr.Markdown("Upload a sustainability report for detailed TCFD analysis")
                
                with gr.Row():
                    file_input = gr.File(
                        label="Upload PDF Report",
                        file_types=[".pdf"],
                        type="filepath"
                    )
                    analyze_btn = gr.Button("Start Analysis", variant="primary")
                
                with gr.Row():
                    progress = gr.Markdown("Upload a report to begin analysis")
                
                with gr.Row():
                    current_question = gr.Markdown("## Current Question: Not started")
                
                with gr.Row():
                    analysis_results = gr.Markdown("Results will appear here...")
            
            # Questions Tab
            with gr.Tab("Questions"):
                gr.Markdown("## TCFD Questions")
                gr.Markdown("Select which questions to include in the analysis")
                
                with gr.Row():
                    select_all = gr.Button("Select All", variant="secondary")
                    clear_all = gr.Button("Clear All", variant="secondary")
                
                questions = service.analyzer.questions["TCFD Analysis"]["questions"]
                checkboxes = []
                
                for i, question in enumerate(questions, 1):
                    checkbox = gr.Checkbox(
                        label=f"Q{i}: {question}",
                        value=True,  # Default to selected
                        interactive=True
                    )
                    checkboxes.append(checkbox)

        async def process_analysis(file, *selected_questions):
            formatted_results = "# Analysis Results\n\n"
            
            if not file:
                yield "Please upload a file first", "## Current Question: Not started", "Waiting for file..."
                return

            try:
                # Convert checkbox selections to question IDs (1-based indexing)
                selected_ids = [i + 1 for i, selected in enumerate(selected_questions) if selected]
                logger.info(f"Selected question IDs: {selected_ids}")
                
                if not selected_ids:
                    yield "Please select at least one question", "## No questions selected", "Select questions to analyze"
                    return
                
                total_steps = len(selected_ids)
                current_step = 0
                
                async for result in service.process_document(file, selected_ids):
                    if "error" in result:
                        yield (
                            f"Error: {result['error']}", 
                            "## Error occurred", 
                            f"Error: {result['error']}"
                        )
                        return
                    
                    if "status" in result:
                        progress_tracker(0, desc=result["status"])
                        continue
                        
                    # Always use the section name from the result, with TCFD Analysis as fallback
                    section = result.get("section", "TCFD Analysis")
                    question_num = result.get("question_number", 0)
                    total_questions = result.get("total_questions", len(selected_ids))
                    
                    logger.info(f"Processing result from {section} question {question_num}/{total_questions}")
                    
                    try:
                        analysis = json.loads(result["result"])
                        logger.info(f"Analysis for {section} Q{question_num}: Score={analysis.get('score', 'N/A')}")
                        logger.info(f"Answer: {analysis.get('answer', 'No answer')[:100]}...")
                    except json.JSONDecodeError:
                        analysis = {
                            "answer": result["result"],
                            "score": "N/A",
                            "evidence": [],
                            "gaps": []
                        }
                        logger.info(f"Raw result for {section} Q{question_num}: {result['result'][:100]}...")
                    
                    formatted_results += f"\n### Question {question_num}\n"
                    formatted_results += f"**Q:** {result['question']}\n\n"
                    formatted_results += f"**A:** {analysis.get('answer', 'No answer provided')}\n\n"
                    if analysis.get('score') != 'N/A':
                        formatted_results += f"**Score:** {analysis.get('score')}\n\n"
                    if analysis.get('evidence'):
                        formatted_results += "**Evidence:**\n" + "\n".join([f"- {e}" for e in analysis['evidence']]) + "\n\n"
                    if analysis.get('gaps'):
                        formatted_results += "**Gaps:**\n" + "\n".join([f"- {g}" for g in analysis['gaps']]) + "\n\n"
                    
                    current_step = question_num
                    progress = current_step / total_questions
                    
                    yield (
                        f"Analyzing {section}: Question {question_num}/{total_questions}",
                        f"## Current Question:\n{result['question']}",
                        formatted_results
                    )
                    progress_tracker(progress, desc=f"Processing question {question_num}/{total_questions}")
                
                progress_tracker(1.0, desc="Complete!")
                logger.info("Analysis complete - Final results compiled")
                yield "✓ Analysis Complete!", "## Analysis Complete", formatted_results
                
            except Exception as e:
                logger.error(f"Error in analysis: {str(e)}", exc_info=True)
                yield (
                    f"Error: {str(e)}",
                    "## Error occurred",
                    f"Error occurred during analysis: {str(e)}"
                )

        def select_all_questions():
            """Select all questions"""
            logger.info("Selecting all questions")
            return [True] * len(checkboxes)
            
        def clear_all_questions():
            """Clear all question selections"""
            logger.info("Clearing all questions")
            return [False] * len(checkboxes)

        # Connect the buttons
        analyze_btn.click(
            fn=process_analysis,
            inputs=[file_input] + checkboxes,
            outputs=[progress, current_question, analysis_results],
            api_name="analyze"
        )
        
        select_all.click(
            fn=select_all_questions,
            inputs=[],
            outputs=checkboxes
        )
        
        clear_all.click(
            fn=clear_all_questions,
            inputs=[],
            outputs=checkboxes
        )

    return app

if __name__ == "__main__":
    app = create_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        debug=True
    ) 