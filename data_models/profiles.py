from dataclasses import dataclass
from typing import List

@dataclass
class Profile:
    """A dataclass representing a user profile for a specific generation."""
    name: str           # The generation name (e.g., "Gen Z")
    birth_years: str    # The birth year range (e.g., "1996–2015")
    description: str    # A brief description of the generation's characteristics

# List of generational profiles
PROFILES: List[Profile] = [
    Profile(
        name="Gen Z",
        birth_years="1996–2015",
        description="Digital natives, environmentally conscious, value affordability."
    ),
    Profile(
        name="Millennials",
        birth_years="1981–1995",
        description="Tech-savvy, socially and environmentally aware, value technology and eco-friendly options."
    ),
    Profile(
        name="Gen X",
        birth_years="1965–1980",
        description="Independent, practical, value work-life balance, often with families needing larger vehicles."
    ),
    Profile(
        name="Baby Boomers",
        birth_years="1946–1964",
        description="Value quality and brand reputation, traditional in choices, may have more disposable income for luxury."
    )
]