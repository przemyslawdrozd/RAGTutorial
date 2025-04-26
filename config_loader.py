import os
import yaml

PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "prompts")


def load_config(path="config.yaml"):
    with open(path, "r") as file:
        return yaml.safe_load(file)


def load_prompt(filename):
    path = os.path.join(PROMPTS_DIR, filename)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


# 🚀 Load prompts
SQL_PROMPT_TEMPLATE = load_prompt("sql_prompt.txt")
PDF_PROMPT_TEMPLATE = load_prompt("pdf_prompt.txt")
