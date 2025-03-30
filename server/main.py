# main.py
import asyncio
from fastapi import FastAPI, WebSocket
from pydantic_models.chat_body import ChatBody
from services.llm_service import LLMService
from services.sort_source_service import SortSourceService
from services.search_service import SearchService
from loguru import logger

app = FastAPI()

search_service = SearchService()
sort_source_service = SortSourceService()
llm_service = LLMService()

@app.websocket("/ws/chat")
async def websocket_chat_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        data = await websocket.receive_json()
        query = data.get("query")
        # Await asynchronous web search and sorting
        search_results = await search_service.web_search(query)
        sorted_results = await sort_source_service.sort_sources(query, search_results)
        await websocket.send_json({"type": "search_result", "data": sorted_results})
        # Generate response (if llm_service.generate_response supports async streaming, adjust accordingly)
        for chunk in llm_service.generate_response(query, sorted_results):
            await websocket.send_json({"type": "content", "data": chunk})
    except Exception as e:
        logger.error(f"Unexpected error occurred in websocket: {e}")
    finally:
        await websocket.close()

@app.post("/chat")
async def chat_endpoint(body: ChatBody):
    try:
        search_results = await search_service.web_search(body.query)
        sorted_results = await sort_source_service.sort_sources(body.query, search_results)
        # If llm_service.generate_response can be made async, await its response; else consider wrapping it.
        response_chunks = []
        for chunk in llm_service.generate_response(body.query, sorted_results):
            response_chunks.append(chunk)
        return {"response": "".join(response_chunks)}
    except Exception as e:
        logger.error(f"Unexpected error occurred in chat endpoint: {e}")
        return {"error": "An error occurred"}
