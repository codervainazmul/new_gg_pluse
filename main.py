from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from g4f.client import Client
import asyncio
import json

app = FastAPI()

# Allow CORS from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = Client()

class PromptRequest(BaseModel):
    model: str
    prompt: str

@app.post("/chat")
async def chat(request: PromptRequest):
    async def generate():
        try:
            # First, try to determine if this is a reasoning model
            is_reasoning_model = any(keyword in request.model.lower() for keyword in 
                                   ['deepseek-r1', 'r1', 'reasoning', 'think'])
            
            # For reasoning models, try streaming with special handling
            if is_reasoning_model:
                try:
                    # Method for reasoning models - sometimes they need different handling
                    chat_completion = client.chat.completions.create(
                        model=request.model,
                        messages=[{"role": "user", "content": request.prompt}],
                        stream=True,
                        temperature=0.7  # Some reasoning models work better with explicit temperature
                    )
                    
                    buffer = ""
                    chunk_count = 0
                    
                    for chunk in chat_completion:
                        chunk_count += 1
                        try:
                            # Multiple ways to extract content from chunk
                            content = None
                            
                            # Method 1: Standard OpenAI format
                            if hasattr(chunk, 'choices') and len(chunk.choices) > 0:
                                if hasattr(chunk.choices[0], 'delta') and hasattr(chunk.choices[0].delta, 'content'):
                                    content = chunk.choices[0].delta.content
                                # Sometimes content is directly in choice
                                elif hasattr(chunk.choices[0], 'text'):
                                    content = chunk.choices[0].text
                                elif hasattr(chunk.choices[0], 'message') and hasattr(chunk.choices[0].message, 'content'):
                                    content = chunk.choices[0].message.content
                            
                            # Method 2: Direct content attribute
                            elif hasattr(chunk, 'content'):
                                content = chunk.content
                            
                            # Method 3: Text attribute
                            elif hasattr(chunk, 'text'):
                                content = chunk.text
                            
                            # Method 4: String conversion as fallback
                            elif isinstance(chunk, str):
                                content = chunk
                            
                            if content:
                                buffer += content
                                yield content
                                await asyncio.sleep(0.01)
                                
                        except Exception as chunk_error:
                            print(f"Reasoning model chunk error: {chunk_error}")
                            continue
                    
                    # If no chunks were processed, try fallback
                    if chunk_count == 0:
                        raise Exception("No chunks received from reasoning model")
                        
                except Exception as reasoning_error:
                    print(f"Reasoning model streaming failed: {reasoning_error}")
                    # Fallback to non-streaming for reasoning models
                    try:
                        chat_completion = client.chat.completions.create(
                            model=request.model,
                            messages=[{"role": "user", "content": request.prompt}],
                            stream=False
                        )
                        
                        # Extract response and stream it
                        if hasattr(chat_completion, 'choices') and len(chat_completion.choices) > 0:
                            full_response = chat_completion.choices[0].message.content
                            # Stream character by character for smooth effect
                            for char in full_response:
                                yield char
                                await asyncio.sleep(0.02)
                        else:
                            yield "No response received from reasoning model"
                            
                    except Exception as fallback_error:
                        yield f"Reasoning model error: {str(fallback_error)}"
            
            else:
                # Standard streaming for regular models
                chat_completion = client.chat.completions.create(
                    model=request.model,
                    messages=[{"role": "user", "content": request.prompt}],
                    stream=True
                )
                
                for chunk in chat_completion:
                    try:
                        content = None
                        
                        # Standard extraction methods
                        if hasattr(chunk, 'choices') and len(chunk.choices) > 0:
                            delta = chunk.choices[0].delta
                            if hasattr(delta, 'content') and delta.content:
                                content = delta.content
                            elif hasattr(chunk.choices[0], 'text'):
                                content = chunk.choices[0].text
                        
                        # Alternative extraction methods
                        elif hasattr(chunk, 'content'):
                            content = chunk.content
                        elif hasattr(chunk, 'text'):
                            content = chunk.text
                        
                        if content:
                            yield content
                            await asyncio.sleep(0.01)
                            
                    except Exception as chunk_error:
                        print(f"Standard model chunk error: {chunk_error}")
                        continue
                        
        except Exception as e:
            print(f"Overall streaming error: {e}")
            # Final fallback - try one more time with basic approach
            try:
                chat_completion = client.chat.completions.create(
                    model=request.model,
                    messages=[{"role": "user", "content": request.prompt}],
                    stream=False
                )
                
                response_text = str(chat_completion.choices[0].message.content)
                for char in response_text:
                    yield char
                    await asyncio.sleep(0.03)
                    
            except Exception as final_error:
                yield f"Error: Unable to get response from {request.model}. Error: {str(final_error)}"
    
    return StreamingResponse(generate(), media_type="text/plain")
