from dataclasses import dataclass
from typing import List, Dict
from data_models.profiles import Profile  # Import Profile for potential future use

@dataclass
class Narrative:
    """A dataclass representing a narrative theme for a generation."""
    name: str         # The narrative theme (e.g., "Sustainability")
    description: str  # A description of the narrative (e.g., "Eco-friendly features")

# Mapping of generation names to their corresponding narratives
NARRATIVES: Dict[str, List[Narrative]] = {
    "Gen Z": [
        Narrative("Sustainability", "Eco-friendly features, electric or hybrid vehicles."),
        Narrative("Advanced Technology", "Connectivity, smart features, and digital integration."),
        Narrative("Affordability", "Cost-effective options, lower price points.")
    ],
    "Millennials": [
        Narrative("Innovative Tech", "Advanced infotainment, autonomous features."),
        Narrative("Eco-Friendly", "Hybrid or electric vehicles, sustainable materials."),
        Narrative("Good Value", "Balance of features and cost, long-term savings.")
    ],
    "Gen X": [
        Narrative("Reliability", "Durable and dependable vehicles."),
        Narrative("Safety", "Family-oriented safety features."),
        Narrative("Spaciousness", "Roomy interiors for family needs.")
    ],
    "Baby Boomers": [
        Narrative("Brand Loyalty", "Trusted manufacturers with a strong reputation."),
        Narrative("Comfort", "Luxurious interiors, smooth ride."),
        Narrative("Traditional Values", "Heritage brands, classic designs.")
    ]
}