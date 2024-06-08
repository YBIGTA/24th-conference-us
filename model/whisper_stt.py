import whisper

def transcribe_audio(file_path):
    # cpu로 바꿈
    model = whisper.load_model("base", device="cpu")
    
    # 경로 설정은 해야됨, 근데 S3에서 바로 받아서 쓰는거라 여기 고치는느낌일듯?
    result = model.transcribe(file_path)
    
    # text로 출력하기
    return result["text"]
