# 베이스 이미지로 Python 3.9 사용
FROM python:3.9

# 패키지 업데이트 및 필수 패키지 설치
RUN apt-get update && apt-get install -y \
    curl \
    xz-utils \
    file \
    sudo \
    git \
    make \
    g++ \
    && apt-get clean

# Mecab 설치
RUN curl -LO https://bitbucket.org/eunjeon/mecab-ko/downloads/mecab-0.996-ko-0.9.2.tar.gz \
    && tar zxfv mecab-0.996-ko-0.9.2.tar.gz \
    && cd mecab-0.996-ko-0.9.2 \
    && ./configure \
    && make \
    && make check \
    && make install

# Mecab-ko-dic 설치
RUN curl -LO https://bitbucket.org/eunjeon/mecab-ko-dic/downloads/mecab-ko-dic-2.1.1-20180720.tar.gz \
    && tar zxfv mecab-ko-dic-2.1.1-20180720.tar.gz \
    && cd mecab-ko-dic-2.1.1-20180720 \
    && ./autogen.sh \
    && ./configure \
    && make \
    && make install

# Python용 Mecab 라이브러리 설치
RUN pip install mecab-python3

# 필요한 추가 Python 패키지 설치 (requirements.txt)
COPY model/requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt

# 작업 디렉토리 설정
WORKDIR /app

# 애플리케이션 파일들을 컨테이너로 복사
COPY . /app

# 컨테이너에서 실행될 기본 명령어
CMD ["python", "model/main.py"]
