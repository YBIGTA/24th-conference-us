from fastapi import FastAPI, HTTPException
import boto3
import os
import sys
from config import settings

from feature import feature_extraction

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
        ### 이 사이에 피처 익스트랙터들 넣을거임###
        
        # Return a success message
        return {"message": f"File {file_key} downloaded successfully to {file_location}"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
