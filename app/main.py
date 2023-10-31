import json
import os

import openai
from dotenv import load_dotenv
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


class TextInput(BaseModel):
    """Represents the text input model for Pydantic validation.

    Attributes:
        text_input (str): The text to be checked.
    """

    text_input: str


app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


@app.api_route("/", response_class=HTMLResponse)
def main(request: Request, text_input: str = Form(None)):
    """Render the main HTML template with optional text input.

    Args:
        request (Request): The FastAPI request object.
        text_input (str, optional): The text to be checked. Defaults to None.

    Returns:
        HTMLResponse: Rendered HTML page.
    """
    context = {"request": request, "text": text_input}
    return templates.TemplateResponse("main.html", context)


@app.post("/check", response_class=HTMLResponse)
async def check_grammar(request: Request, text_input: str = Form(...)):
    """Asynchronously check the grammar of the input text using GPT-3.5 Turbo model.

    Args:
        request (Request): The FastAPI request object.
        text_input (str): The text to be checked for grammar.

    Returns:
        TemplateResponse: Rendered HTML page containing grammar check results.
    """
    model = "gpt-3.5-turbo-16k-0613"
    messages = [
        {
            "role": "user",
            "content": text_input,
        }
    ]
    functions = [
        {
            "name": "log_grammar_check",
            "description": "Check the grammar of the input text",
            "parameters": {
                "type": "object",
                "properties": {
                    "grammar_score": {
                        "type": "integer",
                        "description": "The grammar score of the input text, e.g. 85.5, please make sure the score is between 0 and 100",
                    },
                    "changes": {
                        "type": "array",
                        "description": "The list of changes made to the input text, e.g. 1. 'I am' -> 'I'm'",
                        "items": {
                            "type": "string",
                        },
                    },
                    "reasons": {
                        "type": "array",
                        "description": "The list of reasons for the changes made to the input text, e.g. 1. 'I am' -> 'I'm' because 'I am' is informal",
                        "items": {
                            "type": "string",
                        },
                    },
                },
            },
        }
    ]

    try:
        response = openai.ChatCompletion.create(
            model=model, messages=messages, functions=functions, function_call="auto"
        )
        json_response = json.loads(
            response["choices"][0]["message"]["function_call"]["arguments"]
        )
    except json.JSONDecodeError:
        error_message = "Oops! Something went wrong. Please try again."

    grammar_score = json_response["grammar_score"]
    changes = json_response["changes"]
    reasons = json_response["reasons"]
    error_message = error_message if error_message else None

    context = {
        "request": request,
        "grammar_score": grammar_score,
        "changes": changes,
        "reasons": reasons,
        "error_message": error_message,
    }

    return templates.TemplateResponse("grammar_check_result.html", context)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
