from AudioEmb import feature_ext
from pos import pos
from sbert_embedding import sbert_embedding
from whisper_stt import transcribe_audio

file_location = '/Users/daniel/Desktop/Us/Us/STT/test.m4a'

text = transcribe_audio(file_location)
audio_feature = feature_ext(file_location, text)
pos_feature = pos(text)
sbert_feature = sbert_embedding(text)

result = {
            "audio_feature": audio_feature.tolist(),  # assuming it's a numpy array
            "transcribed_text": text,
            "pos_feature": pos_feature,  # ensure pos_feature is serializable
            "text_embedding": sbert_feature # assuming it's a numpy array
        }
        
        # Return the result with HTTP status 200
        # return {
        #     "status": HTTPStatus.OK,
        #     "message": f"File {file_key} downloaded and processed successfully.",
        #     "features": result
        # }
