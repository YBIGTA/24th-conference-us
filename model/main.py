from fastapi import FastAPI, HTTPException
import boto3
import os
import sys
from config import settings
from http import HTTPStatus
import numpy as np

from AudioEmb import feature_ext
from pos import pos
from sbert_embedding import sbert_embedding
from whisper_stt import transcribe_audio

app = FastAPI()

def get_s3_client():
    return boto3.client(
        's3',
        aws_access_key_id=settings.S3_ACCESS_KEY,
        aws_secret_access_key=settings.S3_SECRET_KEY,
        region_name=settings.REGION
    )

def download_from_s3(local_file_name, s3_bucket, s3_object_key):
    s3 = get_s3_client()
    meta_data = s3.head_object(Bucket=s3_bucket, Key=s3_object_key)
    total_length = int(meta_data.get('ContentLength', 0))
    downloaded = 0

    def progress(chunk):
        nonlocal downloaded
        downloaded += chunk
        done = int(50 * downloaded / total_length)
        sys.stdout.write("\r[%s%s]" % ('=' * done, ' ' * (50-done)))
        sys.stdout.flush()

    print(f'Downloading {s3_object_key}')
    with open(local_file_name, 'wb') as f:
        s3.download_fileobj(s3_bucket, s3_object_key, f, Callback=progress)
    print(f'\nDownloaded {s3_object_key} to {local_file_name}')

@app.get("/FAST/test_s3")
async def test_s3(file_key: str):
    try:
        # Define the local file path for saving the downloaded file
        local_directory = '/Users/daniel/Desktop/Us/data'
        if not os.path.exists(local_directory):
            os.makedirs(local_directory)
        file_location = os.path.join(local_directory, os.path.basename(file_key))
        
        # Download the file from S3
        download_from_s3(file_location, settings.S3_BUCKET_NAME, file_key)
        
        # Logging for debugging
        print("File downloaded successfully. Starting feature extraction...")
        
        # Process the file
        text = transcribe_audio(file_location)
        print("Transcription completed.")
        
        audio_feature = feature_ext(file_location, text)
        print("Audio feature extraction completed.")
        
        pos_feature = pos(text)
        print("POS feature extraction completed.")
        
        sbert_feature = sbert_embedding(text)
        print("SBERT feature extraction completed.")

        # Delete the file after processing
        os.remove(file_location)
        os.remove(file_location[:-3] + 'wav')

        result = {
            "audio_feature": audio_feature.tolist() if isinstance(audio_feature, np.ndarray) else audio_feature,
            "transcribed_text": text,
            "pos_feature": pos_feature if isinstance(pos_feature, list) else list(pos_feature),
            "text_embedding": sbert_feature.tolist() if isinstance(sbert_feature, np.ndarray) else sbert_feature
        }

        # Return the result with HTTP status
        return {
            "status": HTTPStatus.OK,
            "message": f"File {file_key} downloaded and processed successfully.",
            "features": result
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
