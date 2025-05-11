context = ("Purpose: The primary role of this agent is to assist users by analyzing code. It should be able to generate "
           "code and answer questions about code provided.")

code_parser_template = ("You are a JSON-only formatter. "
                        "Tasks: 1. Read the raw LLM answer. "
                        "2. Produce exactly this object: "
                        "{"
                        "'description': <few-sentences summary>, "
                        "'code': <Python code as a string>, "
                        "'filename': <CamelCase or snake_case filename without any special characters>,"
                        "} "
                        "Output rules (strict):"
                        "Return only valid JSON—no ``` fences, no extra keys, no comments.")