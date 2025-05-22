from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from g4f.client import Client

app = FastAPI()

# Allow CORS from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

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
