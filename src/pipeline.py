"""Main pipeline orchestration."""

import time
from typing import Optional
from loguru import logger
from .models import QueryType, AnswerWithSources, ChatMessage
from .parser import parse_file
from .embeddings import VectorStoreManager
from .intent_classifier import IntentClassifier
from .retriever import HybridRetriever
from .llm_interface import ClaudeInterface
from .validator import SpanExtractor, ConfidenceScorer, CitationFormatter
from config.settings import settings
from datetime import datetime


class ExamPrepBot:
    """Main exam prep bot pipeline."""
    
    def __init__(self):
        """Initialize bot components."""
        logger.info("Initializing ExamPrepBot")
        
        self.vector_store_manager = VectorStoreManager()
        self.intent_classifier = IntentClassifier()
        self.retriever = HybridRetriever()
        self.llm = ClaudeInterface()
        self.span_extractor = SpanExtractor()
        self.confidence_scorer = ConfidenceScorer()
        self.chat_history = []
    
    def upload_document(self, file_path: str, file_type: str) -> dict:
        """
        Upload and process document.
        
        Args:
            file_path: Path to document
            file_type: "pdf" or "docx"
            
        Returns:
            Status dict
        """
        logger.info(f"Uploading document: {file_path}")
        start_time = time.time()
        
        try:
            # Parse document
            document = parse_file(file_path)
            
            # Add to vector store
            self.vector_store_manager.add_document(document)
            
            duration = time.time() - start_time
            
            logger.info(f"Document processed in {duration:.2f}s, {document.total_chunks} chunks")
            
            return {
                'success': True,
                'file_name': document.file_name,
                'file_type': document.file_type,
                'total_chunks': document.total_chunks,
                'file_size_mb': document.file_size_bytes / (1024 * 1024),
                'processing_time_seconds': duration,
                'message': f"Successfully uploaded {document.file_name} with {document.total_chunks} chunks"
            }
        
        except Exception as e:
            logger.error(f"Error uploading document: {e}")
            return {
                'success': False,
                'error': str(e),
                'message': f"Failed to upload document: {e}"
            }
    
    def answer_question(self, query: str) -> AnswerWithSources:
        """
        Answer user question.
        
        Args:
            query: User question
            
        Returns:
            AnswerWithSources with answer and citations
        """
        logger.info(f"Processing query: {query}")
        start_time = time.time()
        
        try:
            # Step 1: Classify intent
            intent_result = self.intent_classifier.classify(query)
            intent = intent_result.primary_intent
            intent_confidence = intent_result.confidence
            
            logger.debug(f"Intent: {intent.value}, confidence: {intent_confidence:.2f}")
            
            # Step 2: Retrieve relevant chunks
            retrieval_result = self.retriever.search(query, intent)
            chunks = retrieval_result['chunks']
            is_relevant = retrieval_result['is_relevant']
            relevance_score = retrieval_result['relevance_score']
            
            logger.debug(f"Retrieved {len(chunks)} chunks, in_scope: {is_relevant}")
            
            # Step 3: Check if out of scope
            if not is_relevant:
                logger.warning("Query out of scope")
                return AnswerWithSources(
                    answer="I couldn't find relevant information about this topic in your uploaded materials. Could you rephrase your question or provide more context?",
                    query_intent=intent,
                    intent_confidence=intent_confidence,
                    sources=[],
                    overall_confidence=0.0,
                    hallucination_risk="high",
                    response_time_seconds=time.time() - start_time,
                    format_type="out_of_scope"
                )
            
            # Step 4: Generate answer
            structured_answer = self.llm.generate_structured_answer(query, chunks, intent)
            answer_text = structured_answer['answer']
            claims = structured_answer['claims']
            
            logger.debug(f"Generated answer with {len(claims)} claims")
            
            # Step 5: Extract citations for each claim
            citations = []
            for claim in claims:
                citation = self.span_extractor.extract_supporting_span(claim, chunks)
                if citation:
                    citations.append(citation)
            
            logger.debug(f"Extracted {len(citations)} citations")
            
            # Step 6: Calculate confidence
            overall_confidence = self.confidence_scorer.calculate_answer_confidence(
                chunks, citations, len(claims)
            )
            hallucination_risk = self.confidence_scorer.assess_hallucination_risk(
                chunks, citations, len(claims)
            )
            
            logger.debug(f"Confidence: {overall_confidence:.2f}, Hallucination risk: {hallucination_risk}")
            
            # Step 7: Create response
            response_time = time.time() - start_time
            
            answer_with_sources = AnswerWithSources(
                answer=answer_text,
                query_intent=intent,
                intent_confidence=intent_confidence,
                sources=citations,
                overall_confidence=overall_confidence,
                hallucination_risk=hallucination_risk,
                response_time_seconds=response_time,
                format_type=structured_answer.get('format_type', 'general')
            )
            
            # Step 8: Add to chat history
            self.chat_history.append(
                ChatMessage(
                    role="user",
                    content=query,
                    timestamp=datetime.now().isoformat(),
                    intent_type=intent
                )
            )
            self.chat_history.append(
                ChatMessage(
                    role="assistant",
                    content=answer_text,
                    timestamp=datetime.now().isoformat(),
                    intent_type=intent
                )
            )
            
            logger.info(f"Query processed in {response_time:.2f}s")
            
            return answer_with_sources
        
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            response_time = time.time() - start_time
            
            return AnswerWithSources(
                answer=f"An error occurred: {str(e)}",
                query_intent=QueryType.VAGUE,
                intent_confidence=0.0,
                sources=[],
                overall_confidence=0.0,
                hallucination_risk="high",
                response_time_seconds=response_time,
                format_type="error"
            )
    
    def get_stats(self) -> dict:
        """Get bot statistics."""
        return {
            'vector_store': self.vector_store_manager.get_stats(),
            'chat_history_length': len(self.chat_history),
            'model': settings.model_name,
            'embedding_model': settings.embedding_model
        }
    
    def reset(self) -> None:
        """Reset bot state."""
        logger.info("Resetting bot")
        self.chat_history = []
        self.vector_store_manager = VectorStoreManager()


def create_bot() -> ExamPrepBot:
    """Factory function to create bot instance."""
    return ExamPrepBot()
