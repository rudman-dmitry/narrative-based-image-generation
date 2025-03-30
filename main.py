# main.py
from typing import Dict, Any, Optional
from langgraph.graph import StateGraph, END
from agent.agent_states import AgentState
from agent.agent_nodes import generate_initial_prompt, generate_image, evaluate_image, refine_prompt
import os
from dotenv import load_dotenv
import openai
import requests
from io import BytesIO
from PIL import Image
import time
from data_models.profiles import PROFILES
from data_models.narratives import NARRATIVES
import random
from transformers import CLIPProcessor, CLIPModel
from utils.generate_prompt import generate_image_prompt
from agent.agent_nodes import generate_image_from_prompt, evaluate_image

# Load environment variables
load_dotenv()

# Initialize OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Load CLIP model and processor once at startup
print("Loading CLIP model for evaluation...")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Define the LangGraph workflow
def build_graph() -> StateGraph:
    """Build the LangGraph workflow for image generation and evaluation."""
    graph = StateGraph(AgentState)

    # Add nodes to the workflow
    graph.add_node("generate_prompt", generate_initial_prompt)
    graph.add_node("generate_image", generate_image)
    graph.add_node("evaluate_image", evaluate_image)
    graph.add_node("refine_prompt", refine_prompt)

    # Define the edges between nodes
    graph.add_edge("generate_prompt", "generate_image")
    graph.add_edge("generate_image", "evaluate_image")
    graph.add_conditional_edges(
        "evaluate_image",
        lambda state: "refine" if not state.accepted and state.attempts < state.max_attempts else "stop",
        {"refine": "refine_prompt", "stop": END}
    )
    graph.add_edge("refine_prompt", "generate_image")

    # Set the entry point of the workflow
    graph.set_entry_point("generate_prompt")

    return graph

# Compile the agent from the defined graph
agent = build_graph().compile()

# Function to run the agent
def run_agent(profile_name: str, narrative_name: Optional[str] = None, 
              max_attempts: int = 3, threshold: float = 25.0) -> Dict[str, Any]:
    """Run the agent to generate, evaluate, and refine an image."""
    try:
        # Initialize the agent state with provided parameters
        initial_state = AgentState(
            profile_name=profile_name,
            narrative_name=narrative_name,
            max_attempts=max_attempts,
            threshold=threshold
        )
        # Run the agent and get the final state
        final_state = agent.invoke(initial_state)
        # Return the results as a dictionary
        return {
            "image": final_state.image,
            "score": final_state.score,
            "accepted": final_state.accepted,
            "attempts": final_state.attempts,
            "prompt": final_state.prompt
        }
    except Exception as e:
        print(f"Error during execution: {str(e)}")
        raise

def evaluate_image(image: Image.Image, profile_name: str, narrative_name: str, model, processor) -> dict:
    profile = next((p for p in PROFILES if p.name == profile_name), None)
    narratives = NARRATIVES.get(profile_name, [])
    narrative = next((n for n in narratives if n.name == narrative_name), None)

    if not profile or not narrative:
        raise ValueError("Invalid profile or narrative")

    # Construct evaluation prompts
    car_prompt = f"A car advertisement showcasing a {profile.description.lower()} vehicle with {narrative.description.lower()}."
    style_prompt = "A car advertisement image with modern, high-quality photography, vibrant colors, natural lighting, and a clean background."
    mood_prompt = f"A car advertisement that feels aspirational yet approachable, with lifestyle elements for {profile_name}."

    # Function to compute CLIP score
    def get_clip_score(prompt):
        inputs = processor(text=[prompt], images=image, return_tensors="pt", padding=True, truncation=True, max_length=77)
        outputs = model(**inputs)
        return outputs.logits_per_image.item()

    # Compute scores
    car_score = get_clip_score(car_prompt)
    style_score = get_clip_score(style_prompt)
    mood_score = get_clip_score(mood_prompt)
    average_score = (car_score + style_score + mood_score) / 3

    # Return scores as a dictionary
    return {
        "car": car_score,
        "style": style_score,
        "mood": mood_score,
        "average": average_score
    }

def generate_image_prompt(profile_name: str, narrative_name: str = None) -> str:
    """Generate a detailed prompt for image generation based on a profile and narrative."""
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

def generate_image(prompt: str, max_retries: int = 3) -> Image.Image:
    """Generate an image using DALL-E 3 with simple retry logic."""
    for attempt in range(max_retries):
        try:
            print(f"\nAttempt {attempt + 1}/{max_retries}")
            print("Generating image...")
            
            response = openai.images.generate(
                model="dall-e-3",
                prompt=prompt,
                n=1,
                size="1024x1024",
                quality="standard",
                style="vivid"
            )
            
            print("Image generated, downloading...")
            image_url = response.data[0].url
            image_data = requests.get(image_url).content
            image = Image.open(BytesIO(image_data))
            
            # Save the image immediately after generation
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"generated_car_{timestamp}.png"
            image.save(filename)
            print(f"\nImage saved as '{filename}'")
            
            print("Success!")
            return image
            
        except openai.RateLimitError as e:
            if attempt < max_retries - 1:
                wait_time = 60 * (attempt + 1)  # 60s, 120s, 180s
                print(f"\nRate limit hit. Waiting {wait_time} seconds...")
                time.sleep(wait_time)
                continue
            raise
            
        except Exception as e:
            print(f"\nError: {str(e)}")
            if attempt < max_retries - 1:
                print("Retrying...")
                continue
            raise

def main():
    # Generate prompt for Millennials with Innovative Tech narrative
    profile = "Millennials"
    narrative = "Innovative Tech"
    
    print(f"\nGenerating prompt for {profile} with {narrative} narrative...")
    prompt = generate_image_prompt(profile, narrative)
    print("\nGenerated Prompt:")
    print(prompt)
    
    # Generate image
    print("\nGenerating image...")
    image = generate_image_from_prompt(prompt)
    
    # Save the image
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    image_filename = f"generated_car_{timestamp}.png"
    image.save(image_filename)
    print(f"\nImage saved as {image_filename}")
    
    # Evaluate image
    print("\nEvaluating image...")
    evaluation_results = evaluate_image(image, profile, narrative, model, processor)
    
    print("\nEvaluation Results:")
    for category, score in evaluation_results.items():
        print(f"{category}: {score:.1f}%")

if __name__ == "__main__":
    main()