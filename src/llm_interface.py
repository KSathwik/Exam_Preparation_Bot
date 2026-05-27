"""Anthropic Claude API interface."""

import json
from typing import Optional, List
from loguru import logger
import anthropic
from .models import QueryType, SourceCitation, RetrievedChunk
from .intent_classifier import IntentClassifier
from config.settings import settings


class ClaudeInterface:
    """Interface for interacting with Claude API."""
    
    def __init__(self):
        """Initialize Claude interface."""
        logger.info("Initializing Claude API interface")
        
        self.client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        self.model = settings.model_name
        self.max_tokens = settings.max_tokens
        self.temperature = settings.temperature
        self.top_p = settings.top_p
        self.intent_classifier = IntentClassifier()
    
    def generate_answer(
        self,
        query: str,
        retrieved_chunks: List[RetrievedChunk],
        intent: QueryType
    ) -> str:
        """
        Generate answer using Claude.
        
        Args:
            query: User query
            retrieved_chunks: Retrieved document chunks
            intent: Query intent
            
        Returns:
            Generated answer string
        """
        logger.debug(f"Generating answer for query: {query}")
        
        # Get system prompt for intent
        system_prompt = self.intent_classifier.get_classification_prompt(intent)
        
        # Format retrieved chunks as context
        context = self._format_context(retrieved_chunks)
        
        # Create user message
        user_message = f"""Based on the following document excerpts, answer this query: "{query}"

DOCUMENT EXCERPTS:
{context}

Remember:
1. Answer ONLY based on the provided excerpts
2. If the answer is not in the excerpts, say so clearly
3. Be accurate and cite the source pages when relevant
4. For the intent type '{intent.value}', structure your answer appropriately"""
        
        try:
            logger.debug(f"Calling Claude API with model: {self.model}")
            
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": user_message}
                ]
            )
            
            answer = response.content[0].text
            logger.debug(f"Claude response received: {len(answer)} characters")
            
            return answer
            
        except anthropic.APIError as e:
            logger.error(f"Claude API error: {e}")
            raise
    
    def extract_claims(self, answer: str) -> List[str]:
        """Extract factual claims from generated answer."""
        logger.debug(f"Extracting claims from answer: {len(answer)} characters")
        
        system_prompt = """You are an expert at extracting factual claims from text.
Extract all factual claims (not opinions) from the provided text.
Return as a JSON array of strings.
Example: ["Photosynthesis occurs in chloroplasts", "CO2 is converted to glucose"]"""
        
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=500,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": f"Extract claims: {answer}"}
                ]
            )
            
            response_text = response.content[0].text
            
            # Parse JSON
            try:
                claims = json.loads(response_text)
                if isinstance(claims, list):
                    logger.debug(f"Extracted {len(claims)} claims")
                    return claims
            except json.JSONDecodeError:
                logger.warning("Failed to parse claims as JSON")
                return []
                
        except anthropic.APIError as e:
            logger.error(f"Error extracting claims: {e}")
            return []
    
    def _format_context(self, chunks: List[RetrievedChunk]) -> str:
        """Format retrieved chunks as context for Claude."""
        context_parts = []
        
        for i, chunk in enumerate(chunks, 1):
            page = chunk.metadata.page_number
            section = chunk.metadata.section_title or "Unknown"
            relevance = chunk.relevance_score
            
            context_parts.append(
                f"[{i}] (Page {page}, Section: {section}, Relevance: {relevance:.2f})\n"
                f"{chunk.content}\n"
            )
        
        return "\n".join(context_parts)
    
    def generate_structured_answer(
        self,
        query: str,
        retrieved_chunks: List[RetrievedChunk],
        intent: QueryType
    ) -> dict:
        """
        Generate structured answer with metadata.
        
        Returns:
            {
                'answer': str,
                'format_type': str,
                'claims': List[str],
                'tokens_used': int
            }
        """
        logger.debug(f"Generating structured answer")
        
        # Generate answer
        answer = self.generate_answer(query, retrieved_chunks, intent)
        
        # Extract claims
        claims = self.extract_claims(answer)
        
        # Determine format type
        format_type = self._get_format_type(intent)
        
        return {
            'answer': answer,
            'format_type': format_type,
            'claims': claims,
            'intent': intent.value
        }
    
    def _get_format_type(self, intent: QueryType) -> str:
        """Get output format type based on intent."""
        formats = {
            QueryType.DEFINITION: "definition",
            QueryType.EXPLAIN: "comprehensive",
            QueryType.COMPARE: "comparison",
            QueryType.PROCESS: "ordered_steps",
            QueryType.EXAMPLE: "examples",
            QueryType.DIAGRAM: "description",
            QueryType.VAGUE: "general",
        }
        return formats.get(intent, "general")


def generate_answer(
    query: str,
    retrieved_chunks: List[RetrievedChunk],
    intent: QueryType
) -> str:
    """Convenience function for answer generation."""
    claude = ClaudeInterface()
    return claude.generate_answer(query, retrieved_chunks, intent)
