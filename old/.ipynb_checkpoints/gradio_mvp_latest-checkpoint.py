import gradio as gr
import whisper
import os
# from main import main, load_rife_model
from main_test import main, load_rife_model
import requests
import json
import base64
from TTS.api import TTS

# -------------------------------
# Preload models for faster processing
# -------------------------------
model = whisper.load_model("medium", device="cuda")
_ = load_rife_model()
# Load TTS model (using VCTK VITS model)
tts = TTS("tts_models/en/vctk/vits").to("cuda")

# TTS speaker mapping: P226 for male, P225 for female
TTS_ROLE_MAPPING = {
    "Male": "p226",
    "Female": "p225"
}

# ASL video generation role list (as in your original configuration)
ROLES_ASL = ["trump", "man", "CBC_2"]

OUTPUT_VIDEO_PATH = os.path.join(os.getcwd(), "output", "output1.mp4")
os.makedirs("output", exist_ok=True)

def generate_asl_video(sentence, role):
    if not sentence.strip():
        return "Please enter a valid sentence!", None
    if role not in ROLES_ASL:
        return "Please select a valid role!", None
    try:
        main(sentence, output_video_path=OUTPUT_VIDEO_PATH, role=role)
        if not os.path.exists(OUTPUT_VIDEO_PATH):
            return "Video generation failed!", None
        return "Video generated successfully!", OUTPUT_VIDEO_PATH
    except Exception as e:
        return f"An error occurred: {str(e)}", None

def process_audio(audio_file, role):
    if audio_file is None:
         return "<marquee>No audio detected.</marquee>", "No video generated because no audio was provided."
    try:
        result = model.transcribe(audio_file, language="en")
    except Exception as e:
        return f"<marquee>An error occurred during transcription: {str(e)}</marquee>", None
    text = result.get("text", "").strip()
    transcribed_text = f"<marquee direction='left' scrollamount='5'>{text}</marquee>"
    status, video_path = generate_asl_video(text, role)
    return transcribed_text, video_path

def process_text(text, role):
    if text is None or not text.strip():
         return "<marquee>Please enter a valid sentence!</marquee>", None
    transcribed_text = f"<marquee direction='left' scrollamount='5'>{text}</marquee>"
    status, video_path = generate_asl_video(text, role)
    return transcribed_text, video_path

def clear_all():
    # Clear audio input, transcribed text and video output
    return gr.Audio.update(value=None), "", ""

def clear_text():
    # Clear text input, displayed text and video output for text tab
    return "", "", ""

def process_sign_video_with_audio(video_file, tts_role):
    """
    1. Process the recorded video with the Sign-Speak API for ASL recognition.
    2. Use the recognized text to generate audio with TTS based on the selected role.
    """
    if video_file is None:
        return "No video captured!", None
    try:
        # Read and encode the video file
        file_size = os.path.getsize(video_file)
        with open(video_file, "rb") as f:
            video_bytes = f.read()
        encoded_video = base64.b64encode(video_bytes).decode("utf-8")
        
        # Build JSON payload for Sign-Speak API
        payload_dict = {
            "payload": encoded_video,
            "single_recognition_mode": False,
            "request_class": "BLOCKING",
            "model_version": "SLR.2.sm"
        }
        payload_json = json.dumps(payload_dict)
        
        url = "https://api.sign-speak.com/recognize-sign"
        headers = {
            "X-api-key": "u9RRQt08SbddRlUkL9H3tH6lkOLnSiNx",  # Ensure you use a valid API key
            "Content-Type": "application/json"
        }
        response = requests.request("POST", url, headers=headers, data=payload_json)
        
        result_json = response.json()
        predictions = result_json.get("prediction", [])
        recognized_text = " ".join([pred.get("prediction", "") for pred in predictions]).strip()
        if not recognized_text:
            recognized_text = "No prediction received."
            
        # Generate audio based on the selected TTS role
        speaker = TTS_ROLE_MAPPING.get(tts_role, "p225")
        audio_file_path = os.path.join(os.getcwd(), "output", "output_sign_audio.wav")
        tts.tts_to_file(text=recognized_text, speaker=speaker, file_path=audio_file_path)
        
        return recognized_text, audio_file_path
    except Exception as e:
        return f"Error during recognition: {str(e)}", None

# Custom CSS for layout: both left and right panels and compact audio display
css = """
#custom-container {
    display: flex;
    justify-content: space-around;
    margin: 20px;
}
.audio-short audio {
    max-height: 50px;
}
"""

with gr.Blocks(css=css, elem_id="custom-container") as demo:
    with gr.Row():
        # Left side: ASL Recognition with TTS (layout adjusted)
        with gr.Column(scale=1):
            gr.Markdown("### ASL Sign Language Recognizer")
            gr.Markdown("Record your sign language video using the webcam. After submission, the recognized text will be displayed and converted to audio.")
            sign_video_input = gr.Video(sources=["webcam"], label="Record ASL Video")
            with gr.Row():
                sign_role_dropdown = gr.Dropdown(choices=["Male", "Female"], label="Select TTS Voice", value="Male")
                sign_submit_btn = gr.Button("Submit")
            # Recognized text textbox with reduced height
            sign_recognized_text = gr.Textbox(label="Recognized Text", lines=2)
            # Audio output with compact display (via CSS class "audio-short")
            sign_audio_output = gr.Audio(label="Generated Audio", autoplay=True, elem_classes="audio-short")
            sign_submit_btn.click(
                fn=process_sign_video_with_audio,
                inputs=[sign_video_input, sign_role_dropdown],
                outputs=[sign_recognized_text, sign_audio_output]
            )
        
        # Right side: ASL Sign Language Generator module with both Text and Voice input
        with gr.Column(scale=1):
            gr.Markdown("### ASL Sign Language Generator")
            with gr.Tabs():
                with gr.Tab("Text Input"):
                    gr.Markdown("Enter your text input to generate an ASL video.")
                    text_input = gr.Textbox(label="Enter Text", placeholder="Type your sentence here...", lines=2)
                    role_dropdown_text = gr.Dropdown(ROLES_ASL, label="Select Video Role", value="CBC_2")
                    with gr.Row():
                        submit_btn_text = gr.Button("Submit Text")
                        clear_btn_text = gr.Button("Clear Text")
                    transcribed_text_text = gr.HTML(label="Displayed Text")
                    video_output_text = gr.Video(label="Generated Video", height=360, autoplay=True, loop=True)
                    submit_btn_text.click(
                        fn=process_text,
                        inputs=[text_input, role_dropdown_text],
                        outputs=[transcribed_text_text, video_output_text]
                    )
                    clear_btn_text.click(fn=clear_text, outputs=[text_input, transcribed_text_text, video_output_text])
                
                with gr.Tab("Voice Input"):
                    gr.Markdown("Speak into the microphone. Your speech will be transcribed and used to generate an ASL video based on the selected role.")
                    video_output_voice = gr.Video(label="Generated Video", height=360, autoplay=True, loop=True)
                    transcribed_text_voice = gr.HTML(label="Transcribed Text")
                    with gr.Row():
                        audio_input = gr.Audio(sources=["microphone"], type="filepath", label="Microphone Audio")
                        role_dropdown_voice = gr.Dropdown(ROLES_ASL, label="Select Video Role", value="CBC_2")
                    with gr.Row():
                        submit_btn_voice = gr.Button("Submit")
                        clear_btn_voice = gr.Button("Clear")
                    submit_btn_voice.click(
                        fn=process_audio,
                        inputs=[audio_input, role_dropdown_voice],
                        outputs=[transcribed_text_voice, video_output_voice]
                    )
                    clear_btn_voice.click(fn=clear_all, outputs=[audio_input, transcribed_text_voice, video_output_voice])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=8005, share=True)
