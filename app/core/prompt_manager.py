import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage, SystemMessage
import json
import yaml

class PromptManager:
    def __init__(self, prompts_dir: str = "prompts"):
        self.prompts_dir = Path(prompts_dir)

    def get_analysis_messages(self, question: str, context: str, guidelines: str) -> list:
        """Generate messages for TCFD question analysis"""
        return [
            {
                "role": "system",
                "content": "You are an AI assistant in the role of a Senior Equity Analyst with expertise in climate science that analyzes companys' sustainability reports."
            },
            {
                "role": "user",
                "content": f"""With the following extracted components of the sustainability report at hand, please analyze the question and provide a comprehensive response with evidence and scoring.

QUESTION: {question}
=========
{context}
=========

Please adhere to the following guidelines in your answer:
1. Your response must be precise, thorough, and grounded on specific extracts from the report to verify its authenticity.
2. If you are unsure, simply acknowledge the lack of knowledge, rather than fabricating an answer.
3. Keep your ANSWER within 60 words.
4. Be skeptical to the information disclosed in the report as there might be greenwashing (exaggerating the firm's environmental responsibility). Always answer in a critical tone.
5. cheap talks are statements that are costless to make and may not necessarily reflect the true intentions or future actions of the company. Be critical for all cheap talks you discovered in the report.
6. Always acknowledge that the information provided is representing the company's view based on its report.
7. Scrutinize whether the report is grounded in quantifiable, concrete data or vague, unverifiable statements, and communicate your findings.
{guidelines}

IMPORTANT: Your response MUST be in valid JSON format like this example:
{{
    "ANSWER": "Your detailed analysis of the question",
    "SCORE": "Score from 0-10 indicating disclosure quality and completeness",
    "EVIDENCE": ["List of specific evidence from the text that supports your analysis"],
    "GAPS": ["List of missing information or gaps in disclosure"],
    "SOURCES": [1, 2, 3]  # List of source numbers referenced
}}

Your FINAL_ANSWER in JSON (ensure there's no format error):"""
            }
        ]

    def process_result(self, result: dict, results: dict, q_id: str):
        try:
            result_json = json.loads(result['result'])
            results["answers"][q_id] = result_json.get("ANSWER", "No answer provided")
            results["sources"][q_id] = result_json.get("SOURCES", [])
        except Exception as e:
            print(f"Error processing result: {e}")
            results["answers"][q_id] = "Error processing result"
            results["sources"][q_id] = [] 