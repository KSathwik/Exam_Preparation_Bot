"""Query intent classification module."""

from typing import Dict, Optional
from .models import QueryType, IntentClassificationResult
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from loguru import logger
import numpy as np
from config.settings import settings


class IntentClassifier:
    """Classify user query intent into predefined categories."""
    
    # Intent templates for semantic classification
    INTENT_TEMPLATES = {
        QueryType.DEFINITION: [
            "What is the definition of",
            "Define this term",
            "What does this mean",
            "Explain the term",
        ],
        QueryType.EXPLAIN: [
            "Explain this in detail",
            "Tell me more about",
            "Detail me about",
            "Elaborate on",
            "Describe how it works",
        ],
        QueryType.COMPARE: [
            "Compare and contrast",
            "What is the difference between",
            "How are they different",
            "Similarities and differences",
            "Compare this with",
        ],
        QueryType.PROCESS: [
            "What are the steps",
            "How does it work step by step",
            "What is the process",
            "Describe the procedure",
            "What is the sequence",
        ],
        QueryType.EXAMPLE: [
            "Give me an example",
            "Provide an example",
            "Can you show an example",
            "Real world example",
            "Example of this",
        ],
        QueryType.DIAGRAM: [
            "Explain this diagram",
            "What does this diagram show",
            "Describe this visual",
            "Explain the figure",
            "What is this illustration",
        ],
        QueryType.VAGUE: [
            "What is the answer",
            "Answer this",
            "What about this",
            "Tell me about this",
        ],
    }
    
    # Keywords for rule-based classification
    INTENT_KEYWORDS = {
        QueryType.DEFINITION: ["define", "what is", "meaning", "term", "definition"],
        QueryType.EXPLAIN: ["explain", "detail", "elaborate", "describe", "tell me"],
        QueryType.COMPARE: ["compare", "difference", "vs", "versus", "contrast", "similar"],
        QueryType.PROCESS: ["steps", "process", "how does", "procedure", "sequence"],
        QueryType.EXAMPLE: ["example", "instance", "demonstrate", "show", "case study"],
        QueryType.DIAGRAM: ["diagram", "figure", "visual", "illustration", "chart", "graph"],
        QueryType.HOMEWORK: ["question", "answer", "quiz", "test", "exam"],
    }
    
    def __init__(self):
        """Initialize intent classifier."""
        logger.info("Initializing IntentClassifier")
        self.model = SentenceTransformer(settings.intent_embedding_model)
        self.threshold = settings.intent_classification_threshold
        
        # Pre-compute template embeddings
        self._compute_template_embeddings()
    
    def _compute_template_embeddings(self) -> None:
        """Pre-compute embeddings for all intent templates."""
        self.template_embeddings = {}
        
        for intent, templates in self.INTENT_TEMPLATES.items():
            embeddings = self.model.encode(templates, convert_to_numpy=True)
            # Use mean embedding of all templates for this intent
            self.template_embeddings[intent] = np.mean(embeddings, axis=0)
    
    def classify(self, query: str) -> IntentClassificationResult:
        """
        Classify query intent.
        
        Uses hybrid approach:
        1. Rule-based keywords matching (fast)
        2. Semantic similarity (accurate)
        
        Args:
            query: User query string
            
        Returns:
            IntentClassificationResult with intent and confidence
        """
        logger.debug(f"Classifying query: {query}")
        
        # Step 1: Try rule-based classification
        rule_result = self._classify_by_rules(query)
        if rule_result and rule_result['confidence'] > self.threshold:
            logger.debug(f"Rule-based classification: {rule_result}")
            return self._create_result(query, rule_result)
        
        # Step 2: Use semantic classification
        semantic_result = self._classify_by_semantics(query)
        logger.debug(f"Semantic classification: {semantic_result}")
        
        return self._create_result(query, semantic_result)
    
    def _classify_by_rules(self, query: str) -> Optional[Dict]:
        """Rule-based classification using keywords."""
        query_lower = query.lower()
        intent_scores = {}
        
        for intent, keywords in self.INTENT_KEYWORDS.items():
            # Count keyword matches
            matches = sum(1 for keyword in keywords if keyword in query_lower)
            
            if matches > 0:
                # Score based on percentage of keywords matched
                score = matches / len(keywords)
                intent_scores[intent] = score
        
        if not intent_scores:
            return None
        
        # Get highest scoring intent
        best_intent = max(intent_scores, key=intent_scores.get)
        best_score = intent_scores[best_intent]
        
        return {
            'intent': best_intent,
            'confidence': min(best_score, 0.95),  # Cap at 0.95 for rules
            'all_scores': intent_scores,
            'method': 'rules'
        }
    
    def _classify_by_semantics(self, query: str) -> Dict:
        """Semantic classification using embeddings."""
        # Encode query
        query_embedding = self.model.encode(query, convert_to_numpy=True)
        
        # Calculate similarity with each intent
        intent_scores = {}
        
        for intent, template_embedding in self.template_embeddings.items():
            similarity = cosine_similarity(
                query_embedding.reshape(1, -1),
                template_embedding.reshape(1, -1)
            )[0][0]
            intent_scores[intent] = float(similarity)
        
        # Get highest scoring intent
        best_intent = max(intent_scores, key=intent_scores.get)
        best_score = intent_scores[best_intent]
        
        return {
            'intent': best_intent,
            'confidence': float(best_score),
            'all_scores': intent_scores,
            'method': 'semantic'
        }
    
    def _create_result(self, query: str, classification: Dict) -> IntentClassificationResult:
        """Create IntentClassificationResult from classification dict."""
        intent = classification['intent']
        confidence = classification['confidence']
        all_scores = classification.get('all_scores', {})
        
        # Remove the top intent from alternatives
        alternative_intents = {
            k: v for k, v in all_scores.items() 
            if k != intent
        }
        
        return IntentClassificationResult(
            query=query,
            primary_intent=intent,
            confidence=confidence,
            alternative_intents=alternative_intents,
            reasoning=f"Classified using {classification.get('method', 'unknown')} method"
        )
    
    def get_classification_prompt(self, intent: QueryType) -> str:
        """Get system prompt for Claude based on classification."""
        
        prompts = {
            QueryType.DEFINITION: """You are a helpful study assistant. 
The user is asking for a definition. 
Provide a clear, concise definition based ONLY on the provided document.
If the definition is not in the document, say so clearly.
Format: Start with the term, then provide the definition.""",
            
            QueryType.EXPLAIN: """You are a helpful study assistant.
The user is asking for a detailed explanation.
Provide a comprehensive explanation based ONLY on the provided document.
Include key concepts, mechanisms, and how things work together.
Organize your answer logically with clear sections if needed.""",
            
            QueryType.COMPARE: """You are a helpful study assistant.
The user is asking to compare two or more concepts.
Provide a clear comparison based ONLY on the provided document.
Structure your answer as:
- Similarities (if any)
- Key differences
- When to use each
Use a table format if appropriate.""",
            
            QueryType.PROCESS: """You are a helpful study assistant.
The user is asking about a process or steps.
Provide step-by-step instructions based ONLY on the provided document.
Number the steps clearly and explain what happens at each stage.
Format: Step 1: [description]. Step 2: [description]. etc.""",
            
            QueryType.EXAMPLE: """You are a helpful study assistant.
The user is asking for examples.
Provide concrete examples based ONLY on the provided document.
If examples are not in the document, say so clearly.
Explain why each example is relevant.""",
            
            QueryType.DIAGRAM: """You are a helpful study assistant.
The user is asking about a diagram or figure.
Describe what the diagram shows based ONLY on the provided document.
Explain the key elements and what they represent.
Include the figure number and page reference if mentioned.""",
            
            QueryType.VAGUE: """You are a helpful study assistant.
The user's question is unclear.
Based on the provided document content, offer the most relevant information.
If needed, ask for clarification about what specifically they want to know.""",
        }
        
        return prompts.get(intent, prompts[QueryType.EXPLAIN])


def classify_query(query: str) -> IntentClassificationResult:
    """Convenience function for query classification."""
    classifier = IntentClassifier()
    return classifier.classify(query)
