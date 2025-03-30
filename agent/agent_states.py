from dataclasses import dataclass
from typing import Optional
from PIL import Image

@dataclass
class AgentState:
    """State for the LangGraph agent managing image generation and evaluation."""
    profile_name: str
    narrative_name: Optional[str] = None
    prompt: Optional[str] = None
    image: Optional[Image.Image] = None
    score: Optional[float] = None
    attempts: int = 0
    max_attempts: int = 3
    threshold: float = 25.0
    accepted: bool = False