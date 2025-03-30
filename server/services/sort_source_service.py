# sort_source_service.py
from typing import List
from sentence_transformers import SentenceTransformer
import numpy as np
import asyncio
from loguru import logger

class SortSourceService:
    def __init__(self):
        """Initialize SentenceTransformer model."""
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    async def encode_text(self, text: str) -> np.ndarray:
        """Encodes text asynchronously using thread executor."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.embedding_model.encode, text)

    async def sort_sources(self, query: str, search_results: List[dict]):
        """Sorts sources by relevance score based on embedding similarity."""
        try:
            query_embedding = await self.encode_text(query)
            
            # Generate embeddings concurrently
            tasks = [self.encode_text(res["content"]) for res in search_results]
            embeddings = await asyncio.gather(*tasks)

            relevant_docs = []

            for res, res_embedding in zip(search_results, embeddings):
                similarity = float(
                    np.dot(query_embedding, res_embedding)
                    / (np.linalg.norm(query_embedding) * np.linalg.norm(res_embedding))
                )

                res["relevance_score"] = similarity

                # Filter by similarity threshold
                if similarity > 0.3:
                    relevant_docs.append(res)

            # Sort by relevance score descending
            sorted_docs = sorted(relevant_docs, key=lambda x: x["relevance_score"], reverse=True)

            # Return only the top 5 sources
            return sorted_docs[:5]

        except Exception as e:
            logger.error(f"Sort sources error: {e}")
            return []
