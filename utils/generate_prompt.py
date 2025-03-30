from data_models.profiles import PROFILES
from data_models.narratives import NARRATIVES
from typing import List
import random

def generate_image_prompt(profile_name: str, narrative_name: str = None) -> str:
    """
    Generate a detailed prompt for image generation based on a profile and narrative.
    
    Args:
        profile_name: Name of the generation profile (e.g., "Gen Z")
        narrative_name: Optional specific narrative to focus on
    
    Returns:
        A detailed prompt for image generation
    
    Raises:
        ValueError: If the profile is not found, no narratives exist for the profile, 
                    or the specified narrative is not found.
    """
    # Find the profile
    profile = next((p for p in PROFILES if p.name == profile_name), None)
    if not profile:
        raise ValueError(f"Profile '{profile_name}' not found")
    
    # Get narratives for the profile
    narratives = NARRATIVES.get(profile_name, [])
    if not narratives:
        raise ValueError(f"No narratives found for profile '{profile_name}'")
    
    # If no specific narrative is provided, randomly select one
    if not narrative_name:
        narrative = random.choice(narratives)
    else:
        narrative = next((n for n in narratives if n.name == narrative_name), None)
        if not narrative:
            raise ValueError(f"Narrative '{narrative_name}' not found for profile '{profile_name}'")
    
    # Generate a detailed prompt
    prompt = f"""Create a professional car advertisement image targeting {profile.name} ({profile.birth_years}):
- The car should reflect {profile.description}
- Focus on {narrative.name}: {narrative.description}
- Style: Modern, high-quality photography with vibrant colors
- Composition: Professional automotive photography with attention to detail
- Lighting: Natural daylight with dramatic shadows
- Background: Clean, minimalist setting that complements the car
- Mood: Aspirational yet approachable
- Additional elements: Include subtle lifestyle elements that resonate with {profile.name}"""

    return prompt

# Example usage
if __name__ == "__main__":
    # Generate a prompt for Gen Z with Sustainability narrative
    gen_z_prompt = generate_image_prompt("Gen Z", "Sustainability")
    print("Gen Z Sustainability Prompt:")
    print(gen_z_prompt)
    
    # Generate a random prompt for Millennials
    millennial_prompt = generate_image_prompt("Millennials")
    print("\nRandom Millennial Prompt:")
    print(millennial_prompt)