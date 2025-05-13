"""
Provides LLM-powered analysis of user feedback for generated code.
"""
from typing import List, Dict, Any
from collections import defaultdict

from main import query_llm
from prompts import feedback_analysis_prompt, code_improvement_prompt, feedback_categorization_prompt


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
            
        # Prepare feedback for analysis
        feedback_text = "\n".join([
            f"Rating: {entry['rating']}\n"
            f"Comment: {entry['comment']}\n"
            f"Code Description: {entry.get('code_description', 'Not provided')}\n"
            f"Original Prompt: {entry.get('prompt', 'Not provided')}\n"
            "---"
            for entry in feedback_entries if entry.get('comment')
        ])
        
        prompt = feedback_analysis_prompt.format(feedback_text=feedback_text)
        return query_llm(prompt, self.model, response_format="json")

    def generate_improvement_suggestions(self, code: str, feedback: str, prompt: str = "") -> Dict[str, Any]:
        """
        Generate specific suggestions for improving code based on feedback.
        Args: code: The generated code
              feedback: User feedback about the code
              prompt: The original prompt that generated the code
        Returns: Dict containing categorized improvement suggestions
        """
        if not code or not feedback:
            return {
                "suggestions": [],
                "error": "Both code and feedback are required"
            }

        # Format the prompt
        formatted_prompt = code_improvement_prompt.format(
            code=code,
            prompt=prompt or "Not provided",
            feedback=feedback
        )

        # Get suggestions from LLM
        result = query_llm(formatted_prompt, self.model, response_format="json")
        
        # Ensure we have a valid response
        if not isinstance(result, dict) or "suggestions" not in result:
            return {
                "suggestions": [],
                "error": "Invalid response format from LLM"
            }

        # Validate and normalize suggestions
        valid_categories = {"Quality", "Readability", "Performance", "BestPractices"}
        valid_priorities = {"High", "Medium", "Low"}
        
        normalized_suggestions = []
        for suggestion in result.get("suggestions", []):
            # Skip invalid suggestions
            if not all(key in suggestion for key in ["category", "suggestion", "reason", "priority"]):
                continue
                
            # Normalize category and priority
            category = suggestion["category"]
            if category not in valid_categories:
                category = "Quality"
                
            priority = suggestion["priority"]
            if priority not in valid_priorities:
                priority = "Medium"
            
            normalized_suggestions.append({
                "category": category,
                "suggestion": suggestion["suggestion"],
                "reason": suggestion["reason"],
                "priority": priority
            })
        
        if not normalized_suggestions:
            normalized_suggestions.append({
                "category": "Quality",
                "suggestion": "No specific suggestions provided",
                "reason": "The LLM response was not in the expected format",
                "priority": "Medium"
            })
        
        return {"suggestions": normalized_suggestions}

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
                
            prompt = feedback_categorization_prompt.format(
                feedback=entry['comment'],
                prompt=entry.get('prompt', 'Not provided'),
                code=entry.get('code', 'Not provided')
            )
            
            result = query_llm(prompt, self.model, response_format="json_array")
            
            if isinstance(result, list):
                for category in result:
                    categories[category].append({
                        'comment': entry['comment'],
                        'rating': entry['rating'],
                        'code_id': entry['code_id'],
                        'timestamp': entry['timestamp']
                    })
            else:
                categories["Uncategorized"].append({
                    'comment': entry['comment'],
                    'rating': entry['rating'],
                    'code_id': entry['code_id'],
                    'timestamp': entry['timestamp']
                })
                
        return dict(categories)
