import whisper
from pydub import AudioSegment
import os

from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

import warnings
warnings.filterwarnings('ignore')

import ssl
ssl._create_default_https_context = ssl._create_unverified_context


def Make_Text(name):
    files = [
    f"./data/{name}1.m4a",
    f"./data/{name}2.m4a",
    f"./data/{name}3.m4a",
    ]

    openai_api_key = "sk-proj-HNMrLC5SL9il4okFNe2iT3BlbkFJUurTD1wsM4s4DI8HnMOX"
    llm = ChatOpenAI(api_key=openai_api_key, model_name="gpt-4")
    model = whisper.load_model("base")


    cleaning_prompt_template = PromptTemplate(
    input_variables=["text"],
    template="오타를 수정해줘. 정보를 수정하거나 추가하지 말고 맥락에 따라서 오타만 수정해줘. {text}"
    )
    

    clean_chain = LLMChain(llm=llm, prompt=cleaning_prompt_template)
    
    for file_path in files:   # 음성 파일 불러오기 및 변환
        audio = AudioSegment.from_file(file_path)
        wav_file = f"{file_path[:-4]}.wav"
        audio.export(wav_file, format="wav")

        # 음성 파일을 텍스트로 변환
        result = model.transcribe(wav_file)
        text_result = result["text"]

        # 텍스트를 파일로 저장
        with open(f"{file_path[:-4]}.txt", "w", encoding="utf-8") as f:
            f.write(text_result)
    
        original = open(f"{file_path[:-4]}.txt", "r")
        txt =original.read()
        cleaned = clean_chain.run(text=txt)
        with open(f"{file_path[:-4]}.txt", "w", encoding="utf-8") as f:
            f.write(cleaned)
        