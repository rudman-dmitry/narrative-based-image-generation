import os
import time
import random
from flask import Flask, render_template, request, jsonify
from PIL import Image
from utils.generate_prompt import generate_image_prompt
from agent.agent_nodes import generate_image_from_prompt, evaluate_image
from data_models.profiles import PROFILES
from data_models.narratives import NARRATIVES
import openai
from transformers import CLIPModel, CLIPProcessor
from dotenv import load_dotenv
import io
import base64
import math

# Initialize Flask app
app = Flask(__name__)

# Load environment variables
load_dotenv()

# Initialize OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Load CLIP model and processor once at startup
print("Loading CLIP model for evaluation...")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Ensure static directory exists
if not os.path.exists("static"):
    os.makedirs("static")

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

@app.route("/")
def index():
    """Serve the main page with the profile selection interface."""
    return render_template("index.html")

@app.route("/generate", methods=["POST"])
def generate():
    """Handle image generation and evaluation based on selected profile."""
    try:
        data = request.get_json()
        profile = data.get('profile')
        
        if not profile:
            return jsonify({'error': 'No profile selected'}), 400
            
        # Generate detailed prompt using the prompt generator
        prompt = generate_image_prompt(profile)
        
        # Generate image
        image = generate_image_from_prompt(prompt)
        
        # Convert image to base64 for sending to frontend
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        # Evaluate image
        evaluation_results = evaluate_image(image, prompt)
        
        # Print raw scores for debugging
        print(f"\nRaw CLIP Scores for {profile}:")
        for category, score in evaluation_results['raw_scores'].items():
            print(f"{category}: {score:.2f}")
        
        return jsonify({
            'success': True,
            'image': img_str,
            'scores': evaluation_results['normalized_scores'],
            'raw_scores': evaluation_results['raw_scores'],
            'prompt': prompt
        })
        
    except Exception as e:
        print(f"Error in generate route: {str(e)}")  # Add logging
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5002)