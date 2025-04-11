import io
import base64
from google.cloud import speech
from app.config import settings
import os
import tempfile
from google.oauth2 import service_account
import requests
import json
import google.generativeai as genai

class SpeechService:
    def __init__(self):
        self.client = None
        self.initialize_client()
    
    def initialize_client(self):
        """Initialize the speech-to-text client based on provider"""
        # Option 1: Set environment variable (preferred for production)
        # os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/path/to/your-key-file.json"
        
        # Option 2: Explicitly use credentials (good for development)
        credentials_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                                       "credentials", "google-credentials.json")
        
        if settings.STT_PROVIDER == "google":
            if os.path.exists(credentials_path):
                credentials = service_account.Credentials.from_service_account_file(credentials_path)
                self.client = speech.SpeechClient(credentials=credentials)
            else:
                # Fallback to environment variable or provide a clear error
                try:
                    self.client = speech.SpeechClient()
                except Exception as e:
                    print(f"Google Cloud Speech authentication failed: {e}")
                    print("Please set up credentials by following instructions at:")
                    print("https://cloud.google.com/docs/authentication/external/set-up-adc")
                    print("API will continue running but speech features will return empty results")
                    # Set dummy client that will handle gracefully
                    self.client = None
        elif settings.STT_PROVIDER == "gemini":
            try:
                genai.configure(api_key=settings.GEMINI_API_KEY)
                print("Google Gemini API configured successfully")
            except Exception as e:
                print(f"Google Gemini API configuration failed: {e}")
                print("API will continue running but speech features will return empty results")
    
    async def transcribe_audio(self, audio_content, language_code="vi-VN"):
        """
        Transcribe audio content to text
        
        Args:
            audio_content: Base64 encoded audio content
            language_code: Language code (default: Vietnamese)
            
        Returns:
            Transcribed text
        """
        if settings.STT_PROVIDER == "google" and not self.client:
            # If Google is configured but client failed to initialize
            print("Speech client not available. Speech-to-text features are disabled.")
            
            # Try fallback to OpenAI if API key is available
            if hasattr(settings, "OPENAI_API_KEY") and settings.OPENAI_API_KEY:
                print("Falling back to OpenAI Whisper API")
                return await self._transcribe_with_openai(audio_content, language_code)
                
            return "Speech recognition is not available. Please configure API keys."
            
        if settings.STT_PROVIDER == "google":
            return await self._transcribe_with_google(audio_content, language_code)
        elif settings.STT_PROVIDER == "openai":
            return await self._transcribe_with_openai(audio_content, language_code)
        elif settings.STT_PROVIDER == "gemini":
            return await self._transcribe_with_gemini(audio_content, language_code)
        else:
            raise NotImplementedError(f"STT provider {settings.STT_PROVIDER} not implemented")
    
    async def _transcribe_with_google(self, audio_content, language_code):
        """Transcribe audio using Google Cloud Speech-to-Text"""
        try:
            # Decode base64 audio content
            audio_bytes = base64.b64decode(audio_content)
            
            # Create audio object
            audio = speech.RecognitionAudio(content=audio_bytes)
            
            # Configure recognition settings
            config = speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=16000,
                language_code=language_code,
                enable_automatic_punctuation=True,
                model="default"
            )
            
            # Perform synchronous speech recognition
            response = self.client.recognize(config=config, audio=audio)
            
            # Extract transcription
            transcript = ""
            for result in response.results:
                transcript += result.alternatives[0].transcript
            
            return transcript
        
        except Exception as e:
            print(f"Error transcribing audio: {str(e)}")
            return ""
    
    async def _transcribe_with_openai(self, audio_content, language_code):
        """Transcribe audio using OpenAI Whisper API"""
        try:
            # Decode base64 audio content
            audio_bytes = base64.b64decode(audio_content)
            
            # Create a temporary file for the audio
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
                temp_audio.write(audio_bytes)
                temp_audio_path = temp_audio.name
            
            # OpenAI API requires a file upload
            headers = {
                "Authorization": f"Bearer {settings.OPENAI_API_KEY}"
            }
            
            with open(temp_audio_path, "rb") as audio_file:
                files = {
                    "file": audio_file,
                    "model": (None, "whisper-1"),
                }
                if language_code != "auto":
                    files["language"] = (None, language_code.split("-")[0])  # Extract main language code
                
                response = requests.post(
                    "https://api.openai.com/v1/audio/transcriptions",
                    headers=headers,
                    files=files
                )
            
            # Clean up temporary file
            os.unlink(temp_audio_path)
            
            if response.status_code == 200:
                return response.json().get("text", "")
            else:
                print(f"Error from OpenAI API: {response.text}")
                return ""
                
        except Exception as e:
            print(f"Error transcribing audio with OpenAI: {str(e)}")
            return ""
    
    async def _transcribe_with_gemini(self, audio_content, language_code):
        """Transcribe audio using Google Gemini API"""
        try:
            # Decode base64 audio content
            audio_bytes = base64.b64decode(audio_content)
            
            # Create a temporary file for the audio
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
                temp_audio.write(audio_bytes)
                temp_audio_path = temp_audio.name
            
            # Load audio file
            with open(temp_audio_path, "rb") as audio_file:
                audio_data = audio_file.read()
            
            # Clean up temporary file
            os.unlink(temp_audio_path)
            
            # Create a generation config with the proper language
            generation_config = {
                "temperature": 0,
                "top_p": 1,
                "top_k": 1,
                "max_output_tokens": 2048,
            }
            
            # You might need to adjust this prompt based on Gemini's capabilities
            prompt = f"Transcribe the following audio accurately. The language is {language_code}."
            
            # Create the model
            model = genai.GenerativeModel('gemini-pro-vision', generation_config=generation_config)
            
            # Create multipart content
            contents = [
                {"text": prompt},
                {"inline_data": {"mime_type": "audio/wav", "data": base64.b64encode(audio_data).decode('utf-8')}}
            ]
            
            # Generate response
            response = model.generate_content(contents)
            
            return response.text
                
        except Exception as e:
            print(f"Error transcribing audio with Gemini: {str(e)}")
            return ""
    
    async def detect_language(self, audio_content):
        """
        Detect the language of the audio content
        
        Args:
            audio_content: Base64 encoded audio content
            
        Returns:
            Detected language code
        """
        if not self.client and settings.STT_PROVIDER == "google":
            print("Speech client not available. Language detection is disabled.")
            return "vi-VN"  # Return default language
        
        # This is a simplified implementation
        # In a real-world scenario, you might want to use a dedicated language detection service
        
        # Try to transcribe a small sample with different languages
        languages = ["vi-VN", "en-US"]
        best_language = None
        max_confidence = 0
        
        if settings.STT_PROVIDER == "gemini":
            # For Gemini, we'll just return the default language as it can handle multiple languages
            return "vi-VN"
        
        for lang in languages:
            try:
                # Decode base64 audio content
                audio_bytes = base64.b64decode(audio_content)
                
                # Create audio object
                audio = speech.RecognitionAudio(content=audio_bytes[:30000])  # Use first 30 seconds
                
                # Configure recognition settings
                config = speech.RecognitionConfig(
                    encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                    sample_rate_hertz=16000,
                    language_code=lang,
                    enable_automatic_punctuation=True,
                    model="default"
                )
                
                # Perform synchronous speech recognition
                response = self.client.recognize(config=config, audio=audio)
                
                # Check confidence
                if response.results and response.results[0].alternatives:
                    confidence = response.results[0].alternatives[0].confidence
                    if confidence > max_confidence:
                        max_confidence = confidence
                        best_language = lang
            
            except Exception as e:
                print(f"Error detecting language: {str(e)}")
                continue
        
        # Default to Vietnamese if detection fails
        return best_language or "vi-VN"

speech_service = SpeechService() 