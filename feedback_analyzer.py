"""
Provides LLM-powered analysis of user feedback for generated code.
"""
from typing import List, Dict, Any
from collections import defaultdict

from main import query_llm

class FeedbackAnalyzer:
    """Uses LLM to analyze feedback comments and generate insights about code quality and user satisfaction."""
    
    def __init__(self, model: str = "mistral"):
        self.model = model

    def analyze_feedback(self, feedback_entries: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze a collection of feedback entries to generate insights.
        Args: feedback_entries: List of feedback entries from FeedbackManager
        Returns: Dict containing various analysis results
        """
        if not feedback_entries:
            return {"error": "No feedback entries to analyze"}
            
        # Prepare feedback text directly
        prompt = "Analyze the following feedbacks to provide insights and/or summary about the code quality and user satisfaction to the users.\n"
        i = 1
        for entry in feedback_entries:
            if entry.get('comment'):
                prompt += f"- Prompt {i}: {entry.get('prompt')}, Response Generated: {entry.get('code_description')}, Code Generated: {entry.get('code')}, Rating: {entry.get('rating')}, Comment: {entry.get('comment')}, Models Used: {entry.get('code_model')}, {entry.get('chat_model')}\n"
                i += 1
        prompt += "Return a json object in the format of {'common_themes': [], 'areas_for_improvement': [], 'what_users_like': [], 'suggestions': []}"
        return query_llm(prompt, self.model)

    def generate_improvement_suggestions(self, code: str, feedback: str, prompt: str = "") -> Dict[str, Any]:
        """
        Generate specific suggestions for improving code based on feedback.
        Args: code: The generated code
              feedback: User feedback about the code
              prompt: The original prompt that generated the code
        Returns: Dict containing categorized improvement suggestions
        """
        full_prompt = (f"Analyze this response from LLM and provide improvement suggestions:\n"
                      f"ORIGINAL PROMPT:\n"
                      f"{prompt or 'Not provided'}\n"
                      f"RESPONSE:\n"
                      f"{code}\n"
                      f"FEEDBACK GIVEN BY USER:\n"
                      f"{feedback}\n"
                      f"Provide specific, actionable suggestions for improving the code.")
        return query_llm(full_prompt, self.model)

    def categorize_feedback(self, feedback_entries: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """
        Categorize feedback comments into themes.        
        Args: feedback_entries: List of feedback entries
        Returns: Dict mapping categories to lists of feedback
        """
        categories = defaultdict(list)
        
        for entry in feedback_entries:
            if not entry.get('comment'):
                continue
                
            prompt = (f"Categorize this feedback into one of the following categories [Code Quality, Performance, Readability, Documentation, Functionality, Best Practices]\n"
                      f"FEEDBACK ON RESPONSE: {entry['comment']} ,RESPONSE: {entry.get('code', 'Not provided')} ,PROMPT GIVEN: {entry.get('prompt', 'Not provided')} \n"
                      "Return only the category name as a list.")
            result = query_llm(prompt, self.model)
            if isinstance(result, list):
                for category in result:
                    categories[category].append({
                        'comment': entry['comment'],
                        'rating': entry['rating'],
                        'code_id': entry['code_id'],
                        'timestamp': entry['timestamp'][:16]
                    })
            else:
                categories["Uncategorized"].append({
                    'comment': entry['comment'],
                    'rating': entry['rating'],
                    'code_id': entry['code_id'],
                    'timestamp': entry['timestamp'][:16]
                })
                
        return dict(categories)
