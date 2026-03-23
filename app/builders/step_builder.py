"""Progressive step generator."""


def chat_steps():
    return [
        {"step_number": 1, "label": "Understanding your message..."},
    ]


def db_steps():
    return [
        {"step_number": 1, "label": "Understanding your question..."},
        {"step_number": 2, "label": "Analyzing database schema..."},
        {"step_number": 3, "label": "Identifying tables and columns..."},
        {"step_number": 4, "label": "Grounding filter values..."},
        {"step_number": 5, "label": "Generating SQL..."},
        {"step_number": 6, "label": "Executing query..."},
        {"step_number": 7, "label": "Preparing response..."},
    ]


def file_steps():
    return [
        {"step_number": 1, "label": "Receiving file..."},
        {"step_number": 2, "label": "Extracting content..."},
        {"step_number": 3, "label": "Splitting into chunks..."},
        {"step_number": 4, "label": "Creating embeddings..."},
        {"step_number": 5, "label": "Analyzing content..."},
    ]
