# search_service.py
from config.settings import Settings
from tavily import TavilyClient
import trafilatura
import aiohttp
import asyncio
from loguru import logger

settings = Settings()
tavily_client = TavilyClient(api_key=settings.TAVILY_API_KEY)

class SearchService:
    async def fetch_with_retry(self, url: str, session: aiohttp.ClientSession, retries=3, backoff_factor=0.5) -> str:
        """Fetches URL content with retries and exponential backoff."""
        for attempt in range(retries):
            try:
                async with session.get(url, timeout=10) as response:
                    response.raise_for_status()
                    html = await response.text()

                    # Offload the CPU-bound extraction to a thread
                    loop = asyncio.get_event_loop()
                    content = await loop.run_in_executor(None, trafilatura.extract, html)
                    
                    return content or ""
            
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                logger.warning(f"Attempt {attempt + 1} failed for {url}: {e}")
                if attempt < retries - 1:
                    sleep_time = backoff_factor * (2 ** attempt)
                    await asyncio.sleep(sleep_time)
                else:
                    logger.error(f"Failed to fetch {url} after {retries} attempts.")
                    return ""

    async def web_search(self, query: str):
        """Performs web search and fetches content concurrently."""
        results = []
        try:
            response = tavily_client.search(query, max_results=10)
            search_results = response.get("results", [])
            
            async with aiohttp.ClientSession() as session:
                tasks = [self.fetch_with_retry(result.get("url"), session) for result in search_results]
                contents = await asyncio.gather(*tasks)

                for i, result in enumerate(search_results):
                    results.append(
                        {
                            "title": result.get("title", ""),
                            "url": result.get("url", ""),
                            "content": contents[i] if i < len(contents) else ""
                        }
                    )
            return results
        except Exception as e:
            logger.error(f"Web search error: {e}")
            return results
