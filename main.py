from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from g4f.client import Client
import asyncio
from concurrent.futures import ThreadPoolExecutor
import threading

app = FastAPI()

# Allow CORS from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create a thread pool for blocking operations
executor = ThreadPoolExecutor(max_workers=10)

class PromptRequest(BaseModel):
    model: str
    prompt: str

@app.post("/chat")
async def chat(request: PromptRequest):
    async def generate():
        # Create a queue to communicate between threads
        import queue
        content_queue = queue.Queue()
        error_queue = queue.Queue()
        
        def stream_chat():
            try:
                client = Client()  # Create client in the thread
                chat_completion = client.chat.completions.create(
                    model=request.model,
                    messages=[{"role": "user", "content": request.prompt}], 
                    stream=True
                )
                
                for completion in chat_completion:
                    content = completion.choices[0].delta.content or ""
                    if content:
                        content_queue.put(("content", content))
                
                content_queue.put(("done", None))
                
            except Exception as e:
                error_queue.put(str(e))
                content_queue.put(("error", str(e)))
        
        # Start the streaming in a separate thread
        thread = threading.Thread(target=stream_chat)
        thread.daemon = True
        thread.start()
        
        # Yield content as it comes
        while True:
            try:
                # Check for content with a timeout
                item_type, content = content_queue.get(timeout=0.1)
                
                if item_type == "content":
                    yield content
                elif item_type == "done":
                    break
                elif item_type == "error":
                    yield f"Error: {content}"
                    break
                    
            except queue.Empty:
                # Check if thread is still alive
                if not thread.is_alive():
                    # Thread died, check for errors
                    if not error_queue.empty():
                        error = error_queue.get()
                        yield f"Thread Error: {error}"
                    break
                # Continue waiting
                await asyncio.sleep(0.01)
    
    return StreamingResponse(generate(), media_type="text/plain")

# Alternative approach using asyncio.to_thread (Python 3.9+)
@app.post("/chat-v2")
async def chat_v2(request: PromptRequest):
    async def generate():
        try:
            # Run the blocking g4f call in a thread pool
            def sync_stream():
                client = Client()
                return client.chat.completions.create(
                    model=request.model,
                    messages=[{"role": "user", "content": request.prompt}], 
                    stream=True
                )
            
            # Get the completion object
            chat_completion = await asyncio.get_event_loop().run_in_executor(
                executor, sync_stream
            )
            
            # Now iterate through it
            def get_next_chunk(completion_iter):
                try:
                    return next(completion_iter)
                except StopIteration:
                    return None
                except Exception as e:
                    raise e
            
            # Convert to iterator
            completion_iter = iter(chat_completion)
            
            while True:
                # Get next chunk in thread pool
                chunk = await asyncio.get_event_loop().run_in_executor(
                    executor, get_next_chunk, completion_iter
                )
                
                if chunk is None:
                    break
                    
                content = chunk.choices[0].delta.content or ""
                if content:
                    yield content
                    
        except Exception as e:
            yield f"Error: {str(e)}"
    
    return StreamingResponse(generate(), media_type="text/plain")

# Simple approach - collect all then stream
@app.post("/chat-v3")
async def chat_v3(request: PromptRequest):
    async def generate():
        def get_full_response():
            client = Client()
            chat_completion = client.chat.completions.create(
                model=request.model,
                messages=[{"role": "user", "content": request.prompt}], 
                stream=True
            )
            
            full_text = ""
            for completion in chat_completion:
                content = completion.choices[0].delta.content or ""
                full_text += content
            
            return full_text
        
        try:
            # Get the full response in a thread
            full_response = await asyncio.get_event_loop().run_in_executor(
                executor, get_full_response
            )
            
            # Stream it character by character
            for char in full_response:
                yield char
                await asyncio.sleep(0.01)  # Small delay for streaming effect
                
        except Exception as e:
            yield f"Error: {str(e)}"
    
    return StreamingResponse(generate(), media_type="text/plain")

# Test endpoint
@app.get("/test")
async def test():
    return {"message": "Server is running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
