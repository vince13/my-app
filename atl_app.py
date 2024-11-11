"""
Streamlit interface for Image to Story Application
"""

import streamlit as st
import os
from PIL import Image
import tempfile
from image_to_story_app_huggingface_model import image2text, text2story, story2audio

def main():
    st.title("🔔Image to Story Generator")
    st.write("Upload an image and get a story with audio narration!")

    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Create a temporary file to save the uploaded image
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            temp_path = tmp_file.name

        if st.button('Generate Story'):
            with st.spinner('Generating caption from image...'):
                # Generate caption
                caption = image2text(temp_path)
                st.write("**Generated Caption:**", caption)

            with st.spinner('Creating story...'):
                # Generate story
                story = text2story(caption)
                st.write("**Generated Story:**", story)

            with st.spinner('Creating audio narration...'):
                # Generate audio
                output_file = "generated_story.flac"
                audio_success = story2audio(story, output_file)
                
                if audio_success and os.path.exists(output_file):
                    try:
                        # Read the audio file
                        with open(output_file, 'rb') as audio_file:
                            audio_bytes = audio_file.read()
                        
                        if len(audio_bytes) > 0:
                            # Display audio player
                            st.audio(audio_bytes, format='audio/flac')
                            
                            # Add download button
                            st.download_button(
                                label="Download Audio",
                                data=audio_bytes,
                                file_name="story_narration.flac",
                                mime="audio/flac"
                            )
                        else:
                            st.error("Audio file is empty. Please try again.")
                    except Exception as e:
                        st.error(f"Error processing audio: {str(e)}")
                else:
                    st.error("Failed to generate audio. The story might be too long or there might be an issue with the audio service.")

        # Cleanup temporary files
        os.unlink(temp_path)
        if os.path.exists(output_file):
            os.unlink(output_file)

        if 'story_text' not in st.session_state:
            st.session_state.story_text = None

        if st.button("Generate Story and Audio"):
            with st.spinner("Generating story and audio..."):
                # Your existing story generation code...
                st.session_state.story_text = story_text
                
                # Generate audio
                audio_file = generate_audio(story_text)
                if audio_file:
                    st.audio(audio_file)
                    
        # Add new button for regenerating audio only
        if st.session_state.story_text and st.button("Regenerate Audio Only"):
            with st.spinner("Regenerating audio..."):
                audio_file = generate_audio(st.session_state.story_text)
                if audio_file:
                    st.audio(audio_file)

if __name__ == "__main__":
    main()