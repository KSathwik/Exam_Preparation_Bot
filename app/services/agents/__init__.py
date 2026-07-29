"""Multi-agent RAG pipeline: Orchestrator/Retrieval/Knowledge/Reflection/Memory.

Each agent is a thin wrapper over an existing service (`HybridRetriever`,
`_BaseLLM`, the bot's `chat_history`) — no logic is duplicated here, only
coordinated. See the architecture plan for the full design rationale.
"""

from .knowledge_agent import KnowledgeAgent
from .memory_agent import MemoryAgent
from .orchestrator import OrchestratorAgent
from .reflection_agent import ReflectionAgent
from .retrieval_agent import RetrievalAgent

__all__ = [
    "KnowledgeAgent",
    "MemoryAgent",
    "OrchestratorAgent",
    "ReflectionAgent",
    "RetrievalAgent",
]
