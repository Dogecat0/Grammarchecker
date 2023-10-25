import os
import openai
from dotenv import load_dotenv
from fastapi import FastAPI, Form, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
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
    model = "gpt-3.5-turbo-0613"
    messages = [
        {
            "role": "user",
            "content": (
                "Grammar check, then perform three tasks below and follow \
                    the requirements\n"
                "First task, give the original sentence a grammar score\n"
                "Second task, list what have been changed\n"
                "Third task, state the reason why you changed so\n"
                "Requirements:\n"
                "1. The grammar score should be between 0 and 100 with one \
                    decimal in the format like 'score/100', and it cannot be None\n"
                "2. Please do not return the original sentence\n"
                f"Input text:{text_input}"
            )
        }
    ]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
    )
    response_message = response["choices"][0]["message"]
    sections = response_message["content"].split("\n\n")
    grammar_score = sections[0].split(": ")[1]
    changes = sections[1].split("\n")[1:]
    reasons = sections[2].split("\n")[1:]
    
    context = {
        "request": request,
        "grammar_score": grammar_score,
        "changes": changes,
        "reasons": reasons,
    }

    print(sections)
    return templates.TemplateResponse("grammar_check_result.html", context)
