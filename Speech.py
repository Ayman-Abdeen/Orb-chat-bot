import faster_whisper
import pyttsx3

class SpeechProcessing:
    @staticmethod
    def convert_speech_to_text(audio_path: str) -> str:
        """
        Convert speech from an audio file to text using faster-whisper.

        Parameters:
        audio_path (str): Path to the audio file to be transcribed.

        Returns:
        str: Transcribed text from the audio file.
        """
        # Initialize the faster-whisper model
        model = faster_whisper.WhisperModel("medium")
        
        # Transcribe the audio file
        segments, info = model.transcribe(audio_path)
        
        # Combine all segments into a single string
        transcript = " ".join(segment.text for segment in segments)
        
        return transcript

    @staticmethod
    def convert_text_to_speech(text: str, output_audio_path: str):
        """
        Convert text to speech and save it to an audio file using pyttsx3.

        Parameters:
        text (str): Text to be converted to speech.
        output_audio_path (str): Path to save the output audio file.
        """
        # Initialize the text-to-speech engine
        engine = pyttsx3.init()
        
        # Set properties (optional)
        engine.setProperty('rate', 150)    # Speed of speech
        engine.setProperty('volume', 0.9)  # Volume level (0.0 to 1.0)
        
        # Save speech to a file
        engine.save_to_file(text, output_audio_path)
        
        # Run and wait until the speaking is done
        engine.runAndWait()

# Example usage
if __name__ == "__main__":
    # Convert speech to text
    text = SpeechProcessing.convert_speech_to_text('path_to_audio_file.wav')
    print("Transcribed Text:", text)
    
    # Convert text to speech
    SpeechProcessing.convert_text_to_speech(text, 'output_audio_file.wav')
    print("Text to speech conversion completed.")