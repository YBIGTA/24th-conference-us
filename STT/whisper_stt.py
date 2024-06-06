# whisper_stt.py
import whisper
import argparse

def transcribe_audio(file_path, output_path):
    # Load the Whisper model
    model = whisper.load_model("base")
    
    # Transcribe the audio file
    result = model.transcribe(file_path)
    
    # Save the transcribed text to a file
    with open(output_path, 'w') as f:
        f.write(result["text"])
    
    # Return the transcribed text
    return result["text"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transcribe audio to text using Whisper model")
    parser.add_argument("file_path", type=str, help="Path to the audio file to be transcribed")
    parser.add_argument("output_path", type=str, help="Path to save the transcribed text file")
    
    args = parser.parse_args()
    text = transcribe_audio(args.file_path, args.output_path)
    print("Transcribed Text: ", text)
