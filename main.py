from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from g4f.client import Client

app = FastAPI()
client = Client()

class PromptRequest(BaseModel):
    model: str
    prompt: str

@app.post("/chat")
async def chat(request: PromptRequest):
    def generate():
        chat_completion = client.chat.completions.create(
            model=request.model,
            messages=[{"role": "user", "content": request.prompt}],
            stream=True
        )
        for chunk in chat_completion:
            content = chunk.choices[0].delta.content
            if content:
                yield content

    return StreamingResponse(generate(), media_type="text/plain")
