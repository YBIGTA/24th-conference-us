import whisper
import torch

def check_device():
    # Load the Whisper model
    model = whisper.load_model("base")
    
    # Check if GPU is available and being used
    if torch.cuda.is_available():
        device = next(model.parameters()).device
        return f"Using device: {device}"
    else:
        return "Using device: CPU"

if __name__ == "__main__":
    device_info = check_device()
    print(device_info)
