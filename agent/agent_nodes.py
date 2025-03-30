# agent_nodes.py
import random
import requests
import time
from io import BytesIO
from typing import Optional
from PIL import Image
import openai
from transformers import CLIPProcessor, CLIPModel
from data_models.profiles import PROFILES
from data_models.narratives import NARRATIVES
from utils.generate_prompt import generate_image_prompt
from agent.agent_states import AgentState
from dataclasses import dataclass

# Load CLIP model for image evaluation
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def generate_initial_prompt(state: AgentState) -> AgentState:
    """Generate the initial prompt based on profile and narrative."""
    # Find the profile using dot notation
    profile = next((p for p in PROFILES if p.name == state.profile_name), None)
    if not profile:
        raise ValueError(f"Profile '{state.profile_name}' not found")
    
    # Get narratives for the profile and select one
    narratives = NARRATIVES.get(state.profile_name, [])
    if not state.narrative_name:
        narrative = random.choice(narratives)
    else:
        narrative = next((n for n in narratives if n.name == state.narrative_name), None)
        if not narrative:
            raise ValueError(f"Narrative '{state.narrative_name}' not found for profile '{state.profile_name}'")
    
    # Create the prompt using dot notation for profile and narrative attributes
    prompt = f"""Create a professional car advertisement image targeting {profile.name} ({profile.birth_years}):
- The car should reflect {profile.description}
- Focus on {narrative.name}: {narrative.description}
- Style: Modern, high-quality photography with vibrant colors
- Composition: Professional automotive photography with attention to detail
- Lighting: Natural daylight with dramatic shadows
- Background: Clean, minimalist setting that complements the car
- Mood: Aspirational yet approachable
- Additional elements: Include subtle lifestyle elements that resonate with {profile.name}"""
    
    state.prompt = prompt
    return state

def generate_image(state: AgentState) -> AgentState:
    """Generate an image using OpenAI's DALL-E based on the prompt."""
    max_retries = 3
    base_delay = 60  # base delay in seconds
    
    for attempt in range(max_retries):
        try:
            # Add a small random delay between attempts to avoid rate limits
            if attempt > 0:
                delay = base_delay * (2 ** attempt) + random.uniform(1, 5)  # exponential backoff with jitter
                print(f"\nWaiting {delay:.1f} seconds before attempt {attempt + 1}/{max_retries}...")
                time.sleep(delay)
            
            print(f"\nGenerating image (attempt {attempt + 1}/{max_retries})...")
            response = openai.images.generate(
                model="dall-e-3",
                prompt=state.prompt,
                n=1,
                size="1024x1024",
                quality="standard",
                style="vivid"
            )
            image_url = response.data[0].url
            print("Image generated successfully, downloading...")
            image_data = requests.get(image_url).content
            state.image = Image.open(BytesIO(image_data))
            print("Image downloaded and processed successfully!")
            return state
            
        except openai.RateLimitError as e:
            if attempt < max_retries - 1:
                print(f"\nRate limit hit: {str(e)}")
                continue
            raise ValueError(
                f"Rate limit exceeded after {max_retries} attempts. "
                "Please wait a few minutes before trying again."
            )
            
        except openai.BadRequestError as e:
            if "billing_hard_limit_reached" in str(e):
                raise ValueError(
                    "OpenAI billing limit reached. Please:\n"
                    "1. Check your OpenAI account billing status\n"
                    "2. Update your payment method if needed\n"
                    "3. Set up spending limits in your OpenAI account\n"
                    f"Error details: {str(e)}"
                )
            raise
            
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"\nError occurred: {str(e)}")
                continue
            raise ValueError(f"Failed to generate image after {max_retries} attempts: {str(e)}")
    
    raise ValueError(f"Failed to generate image after {max_retries} attempts")

def generate_image_from_prompt(prompt: str, max_retries: int = 3) -> Image.Image:
    """Generate an image using OpenAI's DALL-E based on a string prompt."""
    base_delay = 60  # base delay in seconds
    
    for attempt in range(max_retries):
        try:
            # Add a small random delay between attempts to avoid rate limits
            if attempt > 0:
                delay = base_delay * (2 ** attempt) + random.uniform(1, 5)  # exponential backoff with jitter
                print(f"\nWaiting {delay:.1f} seconds before attempt {attempt + 1}/{max_retries}...")
                time.sleep(delay)
            
            print(f"\nGenerating image (attempt {attempt + 1}/{max_retries})...")
            response = openai.images.generate(
                model="dall-e-3",
                prompt=prompt,
                n=1,
                size="1024x1024",
                quality="standard",
                style="vivid"
            )
            image_url = response.data[0].url
            print("Image generated successfully, downloading...")
            image_data = requests.get(image_url).content
            image = Image.open(BytesIO(image_data))
            print("Image downloaded and processed successfully!")
            return image
            
        except openai.RateLimitError as e:
            if attempt < max_retries - 1:
                print(f"\nRate limit hit: {str(e)}")
                continue
            raise ValueError(
                f"Rate limit exceeded after {max_retries} attempts. "
                "Please wait a few minutes before trying again."
            )
            
        except openai.BadRequestError as e:
            if "billing_hard_limit_reached" in str(e):
                raise ValueError(
                    "OpenAI billing limit reached. Please:\n"
                    "1. Check your OpenAI account billing status\n"
                    "2. Update your payment method if needed\n"
                    "3. Set up spending limits in your OpenAI account\n"
                    f"Error details: {str(e)}"
                )
            raise
            
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"\nError occurred: {str(e)}")
                continue
            raise ValueError(f"Failed to generate image after {max_retries} attempts: {str(e)}")
    
    raise ValueError(f"Failed to generate image after {max_retries} attempts")

def evaluate_image(image: Image.Image, prompt: str) -> dict:
    """Evaluate the image using CLIP against different aspects of the prompt."""
    # Define different aspects to evaluate
    aspects = {
        'car': 'A car in the image',
        'style': 'Modern, high-quality photography with vibrant colors',
        'mood': 'Aspirational yet approachable mood'
    }
    
    scores = {}
    raw_scores = {}  # Store raw scores for debugging
    
    for category, aspect_prompt in aspects.items():
        # Process the image and text with CLIP
        inputs = processor(
            text=[aspect_prompt],
            images=image,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77
        )
        outputs = model(**inputs)
        # Get raw score
        raw_score = float(outputs.logits_per_image.item())
        raw_scores[category] = raw_score
        
        # Normalize score to percentage (CLIP scores typically range from 0 to 30)
        # Using a more appropriate scaling for the actual range
        normalized_score = (raw_score / 30) * 100  # Scale to 0-100 range
        normalized_score = max(0, min(100, normalized_score))  # Clamp to 0-100
        scores[category] = normalized_score
    
    # Calculate average score
    scores['average'] = sum(scores.values()) / len(scores)
    
    # Print raw scores for debugging
    print(f"\nRaw CLIP Scores for {aspect_prompt}:")
    for category, score in raw_scores.items():
        print(f"{category}: {score:.2f}")
    
    return {
        'normalized_scores': scores,
        'raw_scores': raw_scores
    }

def refine_prompt(state: AgentState) -> AgentState:
    """Refine the prompt if the image doesn't meet the criteria."""
    if state.attempts < state.max_attempts:
        state.attempts += 1
        # Add random emphasis to improve the next generation
        emphasis = random.choice(["eco-friendly features", "technology", "safety", "luxury"])
        state.prompt += f"\n- Ensure the image strongly emphasizes {emphasis} relevant to {state.profile_name}."
    else:
        state.accepted = False  # Max attempts reached, reject the process
    return state