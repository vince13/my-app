"""
Image to Story Application using Hugging Face models
This script converts an image to text, generates a story, and creates audio narration.
"""

# pip install transformers pillow requests
# pip install streamlit pillow
import os
import requests
from transformers import pipeline
from PIL import Image

from dotenv import load_dotenv
load_dotenv()


HUGGINGFACE_API_TOKEN = os.getenv('HUGGINGFACE_API_TOKEN')
if not HUGGINGFACE_API_TOKEN:
    raise ValueError("Please set the HUGGINGFACE_API_TOKEN environment variable")


# Initialize models globally to avoid reloading
IMAGE_MODEL = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
STORY_MODEL = pipeline('text-generation', model='gpt2', max_new_tokens=500)

def image2text(image_path: str) -> str:
    """
    Convert image to descriptive text
    Args:
        image_path: Path to the input image
    Returns:
        str: Generated caption for the image
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
        
    try:
        image = Image.open(image_path)
        result = IMAGE_MODEL(image)[0]["generated_text"]
        return result
    except Exception as e:
        print(f"Error generating caption: {str(e)}")
        raise

def text2story(text: str, max_new_tokens: int = 500) -> str:
    """
    Generate a story from the input text
    Args:
        text: Input text to base the story on
        max_new_tokens: Maximum number of new tokens to generate
    Returns:
        str: Generated story
    """
    try:
        generated_story = STORY_MODEL(
            text, 
            max_new_tokens=max_new_tokens,
            num_return_sequences=1,
            truncation=True,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.2
        )
        return generated_story[0]["generated_text"]
    except Exception as e:
        print(f"Error generating story: {str(e)}")
        return ""

def story2audio(story: str, output_file: str = "model_result.flac") -> bool:
    """
    Convert story text to audio
    Args:
        story: The story text to convert to audio
        output_file: Path to save the audio file
    Returns:
        bool: True if audio generation was successful
    """
    api_url = "https://api-inference.huggingface.co/models/espnet/kan-bayashi_ljspeech_vits"
    headers = {"Authorization": f"Bearer {HUGGINGFACE_API_TOKEN}"}

    try:
        story = story.strip()
        if not story:
            raise ValueError("Story text is empty")

        # Try with full story first
        response = requests.post(
            api_url,
            headers=headers,
            json={"inputs": story}
        )

        # If full story fails, try with truncated version
        if response.status_code != 200:
            truncated_story = story[:150]  # Simple truncation if full story fails
            response = requests.post(
                api_url,
                headers=headers,
                json={"inputs": truncated_story}
            )

        if response.status_code == 200:
            with open(output_file, "wb") as file:
                file.write(response.content)
            print(f"Audio saved successfully to {output_file}")
            return True
        else:
            print(f"Audio generation failed with status code {response.status_code}")
            print(f"Response content: {response.text}")
            return False
    except Exception as e:
        print(f"Error generating audio: {str(e)}")
        return False

def main():
    """Main function to demonstrate the image-to-story-to-audio pipeline"""
    # Example usage with your image path
    image_path = "/content/forest1.jpeg"
    
    # Generate caption from image
    caption = image2text(image_path)
    print("Generated Caption:", caption)

    # Generate story from caption
    story = text2story(caption)
    print("\nGenerated Story:", story)

    # Create audio from story
    audio_success = story2audio(story)
    if audio_success:
        print("\nAudio generation completed successfully!")

if __name__ == "__main__":
    main()


# streamlit run atl_app.py




