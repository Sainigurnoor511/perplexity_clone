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


# chat websocket
@app.websocket("/ws/chat")
async def websocket_chat_endpoint(websocket: WebSocket):
    """ WebSocket endpoint to handle chat requests. """

    await websocket.accept()

    try:
        await asyncio.sleep(0.1)
        data = await websocket.receive_json()
        query = data.get("query")
        search_results = search_service.web_search(query)
        sorted_results = sort_source_service.sort_sources(query, search_results)
        await asyncio.sleep(0.1)
        await websocket.send_json({"type": "search_result", "data": sorted_results})
        for chunk in llm_service.generate_response(query, sorted_results):
            await asyncio.sleep(0.1)
            await websocket.send_json({"type": "content", "data": chunk})

    except Exception as e:
        logger.error(f"Unexpected error occurred in websocket: {e}")
    finally:
        await websocket.close()


# chat
@app.post("/chat")
def chat_endpoint(body: ChatBody):
    """ Endpoint to handle chat requests."""
    
    try:
        search_results = search_service.web_search(body.query)
        sorted_results = sort_source_service.sort_sources(body.query, search_results)
        response = llm_service.generate_response(body.query, sorted_results)
        return response

    except Exception as e:
        logger.error(f"Unexpected error occurred in chat endpoint: {e}")
        return None
