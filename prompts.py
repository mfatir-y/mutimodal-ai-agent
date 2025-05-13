context = ("Purpose: The primary role of this agent is to assist users by analyzing code. It should be able to generate "
           "code and answer questions about code provided.")

code_parser_template = ("Parse the response from the previous LLM into a description and a string of valid code. "
                        "also come up with a valid filename that could be saved which doesn't contain any special "
                        "characters. Here is the response: {response}. "
                        "You should parse this in the following JSON format: "
                        "Return only the JSON object.")

feedback_analysis_prompt = """Analyze the following user feedback for generated code and provide insights:
{feedback_text}

Analyze the feedback and return a JSON object with the following structure:
{
    "common_themes": ["theme1", "theme2", ...],
    "areas_for_improvement": ["area1", "area2", ...],
    "user_likes": ["like1", "like2", ...],
    "improvement_suggestions": ["suggestion1", "suggestion2", ...]
}
Focus on actionable insights and specific patterns in the feedback.
Return only the JSON object with no additional text."""

code_improvement_prompt = """You are a code improvement advisor. Your task is to analyze code and feedback to suggest improvements.

INPUT:
CODE:
{code}

PROMPT:
{prompt}

FEEDBACK:
{feedback}

OUTPUT INSTRUCTIONS:
Return a JSON object with this exact structure:
{{
    "suggestions": [
        {{
            "category": "Quality",
            "suggestion": "A specific improvement suggestion",
            "reason": "Why this improvement would help",
            "priority": "High"
        }}
    ]
}}

RULES:
1. category must be one of: Quality, Readability, Performance, BestPractices
2. priority must be one of: High, Medium, Low
3. Use proper JSON format with double quotes
4. Return only the JSON object, no other text
5. Include at least one suggestion
6. Each suggestion must have all four fields"""

feedback_categorization_prompt = """Categorize this feedback into relevant categories:
FEEDBACK: {feedback}
CODE PROMPT: {prompt}
GENERATED CODE: {code}

Return a JSON array containing only the relevant category names from this list:
- Code Quality
- Performance
- Readability
- Documentation
- Functionality
- Best Practices
Each feedback should be categorized into only one of the above categories.
Example response: ["Code Quality", "Readability"]
Return only the JSON array with no additional text."""