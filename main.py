from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from g4f.client import Client
import asyncio
import json
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    def generate():  # Make this sync, not async
        try:
            logger.info(f"Attempting to chat with model: {request.model}")
            logger.info(f"Prompt: {request.prompt}")
            
            # Try streaming first
            try:
                chat_completion = client.chat.completions.create(
                    model=request.model,
                    messages=[{"role": "user", "content": request.prompt}],
                    stream=True
                )
                
                logger.info("Got chat completion object, starting to iterate...")
                chunk_count = 0
                total_content = ""
                
                for chunk in chat_completion:
                    chunk_count += 1
                    logger.info(f"Processing chunk {chunk_count}: {type(chunk)}")
                    
                    content = None
                    
                    try:
                        # Debug: Print the chunk structure
                        logger.info(f"Chunk attributes: {dir(chunk)}")
                        
                        # Try different ways to extract content
                        if hasattr(chunk, 'choices') and chunk.choices:
                            choice = chunk.choices[0]
                            logger.info(f"Choice attributes: {dir(choice)}")
                            
                            if hasattr(choice, 'delta') and choice.delta:
                                if hasattr(choice.delta, 'content') and choice.delta.content:
                                    content = choice.delta.content
                                    logger.info(f"Found delta content: {repr(content)}")
                            elif hasattr(choice, 'text') and choice.text:
                                content = choice.text
                                logger.info(f"Found choice text: {repr(content)}")
                            elif hasattr(choice, 'message') and choice.message and hasattr(choice.message, 'content'):
                                content = choice.message.content
                                logger.info(f"Found message content: {repr(content)}")
                        
                        # Alternative: direct content
                        elif hasattr(chunk, 'content') and chunk.content:
                            content = chunk.content
                            logger.info(f"Found direct content: {repr(content)}")
                        
                        # Alternative: text attribute
                        elif hasattr(chunk, 'text') and chunk.text:
                            content = chunk.text
                            logger.info(f"Found text attribute: {repr(content)}")
                        
                        if content:
                            total_content += content
                            yield content
                        else:
                            logger.warning(f"No content found in chunk {chunk_count}")
                            
                    except Exception as chunk_error:
                        logger.error(f"Error processing chunk {chunk_count}: {chunk_error}")
                        continue
                
                logger.info(f"Finished streaming. Total chunks: {chunk_count}, Total content length: {len(total_content)}")
                
                # If we got no content at all, try the fallback
                if chunk_count == 0 or not total_content.strip():
                    logger.warning("No content received from streaming, trying fallback...")
                    raise Exception("No content received from streaming")
                    
            except Exception as streaming_error:
                logger.error(f"Streaming failed: {streaming_error}")
                logger.info("Trying non-streaming fallback...")
                
                # Fallback to non-streaming
                try:
                    chat_completion = client.chat.completions.create(
                        model=request.model,
                        messages=[{"role": "user", "content": request.prompt}],
                        stream=False
                    )
                    
                    logger.info("Got non-streaming response")
                    
                    # Extract the full response
                    if hasattr(chat_completion, 'choices') and chat_completion.choices:
                        choice = chat_completion.choices[0]
                        if hasattr(choice, 'message') and choice.message and hasattr(choice.message, 'content'):
                            full_response = choice.message.content
                        elif hasattr(choice, 'text'):
                            full_response = choice.text
                        else:
                            full_response = str(choice)
                    else:
                        full_response = str(chat_completion)
                    
                    logger.info(f"Full response length: {len(full_response)}")
                    
                    # Stream it character by character
                    for char in full_response:
                        yield char
                        
                except Exception as fallback_error:
                    logger.error(f"Fallback also failed: {fallback_error}")
                    yield f"Error: Both streaming and non-streaming failed. Model: {request.model}, Error: {str(fallback_error)}"
                    
        except Exception as e:
            logger.error(f"Overall error: {e}")
            yield f"Critical Error: {str(e)}"
    
    return StreamingResponse(generate(), media_type="text/plain")

# Add a simple test endpoint
@app.get("/test")
async def test():
    return {"message": "Server is running"}

# Add a debug endpoint to test model without streaming
@app.post("/debug")
async def debug_model(request: PromptRequest):
    try:
        logger.info(f"Debug testing model: {request.model}")
        
        # Try non-streaming first
        chat_completion = client.chat.completions.create(
            model=request.model,
            messages=[{"role": "user", "content": request.prompt}],
            stream=False
        )
        
        logger.info("Debug: Got response")
        logger.info(f"Debug: Response type: {type(chat_completion)}")
        logger.info(f"Debug: Response attributes: {dir(chat_completion)}")
        
        return {
            "success": True,
            "response_type": str(type(chat_completion)),
            "response_str": str(chat_completion),
            "has_choices": hasattr(chat_completion, 'choices'),
            "choices_len": len(chat_completion.choices) if hasattr(chat_completion, 'choices') else 0
        }
        
    except Exception as e:
        logger.error(f"Debug error: {e}")
        return {
            "success": False,
            "error": str(e)
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
