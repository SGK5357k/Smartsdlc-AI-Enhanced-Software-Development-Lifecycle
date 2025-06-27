# SmartSDLC - AI Language Learning Platform (Fixed Version)
# Deploy this in Google Colab with Gradio

# First, install required packages (run in Colab cell):
# !pip install transformers torch gradio accelerate bitsandbytes sentencepiece

import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import json
import re
from datetime import datetime
import os
import warnings
warnings.filterwarnings("ignore")

class SmartSDLC:
    def __init__(self):
        self.model_name = "ibm-granite/granite-3.3-2b-instruct"
        self.tokenizer = None
        self.model = None
        self.pipeline = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        self.load_model()

    def load_model(self):
        """Load the IBM Granite model and tokenizer with proper error handling"""
        try:
            print("Loading IBM Granite 3.3-2B Instruct model...")

            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                padding_side="left"
            )

            # Add pad token if it doesn't exist
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Load model with proper device handling
            if self.device == "cuda":
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    trust_remote_code=True,
                    low_cpu_mem_usage=True
                )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float32,
                    trust_remote_code=True
                )
                self.model = self.model.to(self.device)

            print("Model loaded successfully!")

        except Exception as e:
            print(f"Error loading model: {e}")
            # Fallback to a simpler model or CPU-only mode
            self.load_fallback_model()

    def load_fallback_model(self):
        """Fallback model loading for when main model fails"""
        try:
            print("Loading fallback model...")
            self.model_name = "microsoft/DialoGPT-medium"  # Smaller fallback model

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float32
            ).to("cpu")  # Force CPU for fallback

            self.device = "cpu"
            print("Fallback model loaded on CPU")

        except Exception as e:
            print(f"Fallback model also failed: {e}")
            self.model = None
            self.tokenizer = None

    def generate_response(self, prompt, max_length=512, temperature=0.7):
        """Generate response with proper error handling"""
        if self.model is None or self.tokenizer is None:
            return "Model not loaded. Please check the setup."

        try:
            # Encode input
            inputs = self.tokenizer.encode(prompt, return_tensors="pt", truncation=True, max_length=400)
            inputs = inputs.to(self.device)

            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_length=max_length,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    num_return_sequences=1,
                    attention_mask=inputs.ne(self.tokenizer.pad_token_id).long()
                )

            # Decode response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Remove the input prompt from response
            if prompt in response:
                response = response.replace(prompt, "").strip()

            return response if response else "I apologize, I couldn't generate a proper response. Please try again."

        except Exception as e:
            return f"Error generating response: {str(e)}. Try with shorter input text."

    def correct_grammar(self, text):
        """Grammar and spelling correction"""
        if not text.strip():
            return "Please enter some text to correct.", ""

        prompt = f"""Please correct the grammar and spelling in this text while maintaining its original meaning:

Text to correct: "{text}"

Corrected text:"""

        corrected = self.generate_response(prompt, max_length=300)

        # Simple explanation generation
        explanation = "Grammar and spelling have been corrected while preserving the original meaning."

        return corrected, explanation

    def detect_language(self, text):
        """Detect the language of input text"""
        if not text.strip():
            return "Please enter some text for language detection."

        prompt = f"""Identify the language of this text and provide confidence level:

Text: "{text}"

Language:"""

        return self.generate_response(prompt, max_length=100)

    def generate_exercise(self, exercise_type, language, difficulty):
        """Generate learning exercises"""
        prompt = f"""Create a {difficulty} level {exercise_type} exercise in {language}:

Exercise Type: {exercise_type}
Language: {language}
Difficulty: {difficulty}

Please provide:
1. Exercise question/prompt
2. Example answer or solution
3. Learning tip

Exercise:"""

        return self.generate_response(prompt, max_length=400)

    def analyze_text(self, text, analysis_type):
        """Analyze text for various aspects"""
        if not text.strip():
            return "Please enter some text to analyze."

        prompt = f"""Perform {analysis_type} analysis on this text:

Text: "{text}"

Analysis ({analysis_type}):"""

        return self.generate_response(prompt, max_length=300)

# Initialize the SmartSDLC instance
print("Initializing SmartSDLC...")
smart_sdlc = SmartSDLC()

# Create Gradio Interface
def create_interface():
    with gr.Blocks(
        theme=gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="slate",
            neutral_hue="slate"
        ),
        css="""
        .gradio-container {
            background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
            color: white;
        }
        .gr-button-primary {
            background: linear-gradient(45deg, #3b82f6, #6366f1) !important;
            border: none !important;
        }
        .gr-textbox, .gr-dropdown {
            background-color: #374151 !important;
            border-color: #4b5563 !important;
            color: white !important;
        }
        """
    ) as demo:

        gr.HTML("""
        <div style="text-align: center; padding: 20px;">
            <h1 style="color: white; font-size: 2.5em; margin-bottom: 10px;">
                🚀 SmartSDLC - AI Language Learning Platform
            </h1>
            <p style="color: #94a3b8; font-size: 1.1em;">
                Powered by IBM Granite 3.3-2B Instruct Model
            </p>
            <p style="color: #64748b;">
                Welcome to your comprehensive AI-powered language learning assistant!
            </p>
        </div>
        """)

        with gr.Tabs():
            # Tab 1: Real-time Corrections
            with gr.Tab("✏️ Real-time Corrections"):
                gr.Markdown("## Grammar and Spelling Correction")

                with gr.Row():
                    with gr.Column():
                        input_text = gr.Textbox(
                            label="Enter text to correct",
                            placeholder="Type your text here...",
                            lines=5
                        )
                        correct_btn = gr.Button("Correct Text", variant="primary")

                    with gr.Column():
                        corrected_output = gr.Textbox(
                            label="Corrected Text",
                            lines=5,
                            interactive=False
                        )
                        explanation_output = gr.Textbox(
                            label="Explanations",
                            lines=3,
                            interactive=False
                        )

                correct_btn.click(
                    smart_sdlc.correct_grammar,
                    inputs=[input_text],
                    outputs=[corrected_output, explanation_output]
                )

            # Tab 2: Language Detection
            with gr.Tab("🌍 Language Detection"):
                gr.Markdown("## Automatic Language Detection")

                detect_input = gr.Textbox(
                    label="Enter text for language detection",
                    placeholder="Enter text in any language...",
                    lines=3
                )
                detect_btn = gr.Button("Detect Language", variant="primary")
                detect_output = gr.Textbox(
                    label="Detection Result",
                    lines=3,
                    interactive=False
                )

                detect_btn.click(
                    smart_sdlc.detect_language,
                    inputs=[detect_input],
                    outputs=[detect_output]
                )

            # Tab 3: Learning Exercises
            with gr.Tab("📚 Learning Exercises"):
                gr.Markdown("## Multilingual Learning Exercises")

                with gr.Row():
                    exercise_type = gr.Dropdown(
                        label="Exercise Type",
                        choices=[
                            "Grammar Practice",
                            "Vocabulary Building",
                            "Reading Comprehension",
                            "Writing Practice",
                            "Conversation Starters",
                            "Pronunciation Guide"
                        ],
                        value="Grammar Practice"
                    )

                    language = gr.Dropdown(
                        label="Language",
                        choices=["English", "Spanish", "French", "German", "Italian", "Portuguese", "Chinese", "Japanese"],
                        value="English"
                    )

                    difficulty = gr.Dropdown(
                        label="Difficulty/Topic",
                        choices=["beginner", "intermediate", "advanced"],
                        value="intermediate"
                    )

                generate_btn = gr.Button("Generate Exercise", variant="primary")
                exercise_output = gr.Textbox(
                    label="Generated Exercise",
                    lines=8,
                    interactive=False
                )

                generate_btn.click(
                    smart_sdlc.generate_exercise,
                    inputs=[exercise_type, language, difficulty],
                    outputs=[exercise_output]
                )

            # Tab 4: Text Analysis
            with gr.Tab("📊 Text Analysis"):
                gr.Markdown("## Advanced Text Analysis")

                analysis_text = gr.Textbox(
                    label="Enter text to analyze",
                    placeholder="Paste your text here for analysis...",
                    lines=5
                )

                analysis_type = gr.Dropdown(
                    label="Analysis Type",
                    choices=[
                        "Sentiment Analysis",
                        "Readability Score",
                        "Complexity Analysis",
                        "Tone Detection",
                        "Key Themes",
                        "Writing Style"
                    ],
                    value="Sentiment Analysis"
                )

                analyze_btn = gr.Button("Analyze Text", variant="primary")
                analysis_output = gr.Textbox(
                    label="Analysis Results",
                    lines=6,
                    interactive=False
                )

                analyze_btn.click(
                    smart_sdlc.analyze_text,
                    inputs=[analysis_text, analysis_type],
                    outputs=[analysis_output]
                )

            # Tab 5: About
            with gr.Tab("ℹ️ About"):
                gr.Markdown("""
                ## About SmartSDLC

                SmartSDLC is an advanced AI-powered language learning platform that leverages the IBM Granite 3.3-2B Instruct model to provide:

                ### Features:
                - **Real-time Grammar Correction**: Instant grammar and spelling fixes with explanations
                - **Language Detection**: Automatic identification of text language
                - **Interactive Exercises**: Customized learning exercises for multiple languages
                - **Text Analysis**: Advanced linguistic analysis including sentiment, tone, and complexity
                - **Multilingual Support**: Support for major world languages

                ### Technology Stack:
                - **Model**: IBM Granite 3.3-2B Instruct
                - **Framework**: Hugging Face Transformers
                - **Interface**: Gradio
                - **Platform**: Google Colab compatible

                ### Usage Tips:
                - For best results, keep input text under 500 words
                - The model works better with clear, complete sentences
                - Try different difficulty levels in exercises to match your skill level

                **Built with ❤️ using IBM Granite AI**
                """)

        gr.HTML("""
        <div style="text-align: center; padding: 20px; border-top: 1px solid #374151; margin-top: 20px;">
            <p style="color: #94a3b8;">
                🔥 Use via API • 😊 Built with Gradio • ⚙️ Settings
            </p>
        </div>
        """)

    return demo

# Launch the application
if __name__ == "__main__":
    print("Creating Gradio interface...")
    demo = create_interface()
    print("Launching SmartSDLC...")

    # Launch with public sharing for Colab
    demo.launch(
        share=True,
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True
    )