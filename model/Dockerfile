# 베이스 이미지
FROM python:3.9

# 필수 패키지 설치
RUN apt-get update && apt-get install -y \
    curl \
    xz-utils \
    file \
    sudo \
    git \
    make \
    g++ \
    ffmpeg \
    automake \
    libtool \
    pkg-config \
    && apt-get clean

# Mecab 설치
RUN cd /tmp \
    && curl -LO https://bitbucket.org/eunjeon/mecab-ko/downloads/mecab-0.996-ko-0.9.2.tar.gz \
    && tar zxfv mecab-0.996-ko-0.9.2.tar.gz \
    && cd mecab-0.996-ko-0.9.2 \
    && ./configure \
    && make \
    && make install

# 라이브러리 경로 갱신
RUN ldconfig

# Mecab-ko-dic 설치
RUN cd /tmp \
    && curl -LO https://bitbucket.org/eunjeon/mecab-ko-dic/downloads/mecab-ko-dic-2.1.1-20180720.tar.gz \
    && tar zxvf mecab-ko-dic-2.1.1-20180720.tar.gz \
    && cd mecab-ko-dic-2.1.1-20180720 \
    && ./autogen.sh \
    && ./configure \
    && make \
    && sh -c 'echo "dicdir=/usr/local/lib/mecab/dic/mecab-ko-dic" > /usr/local/etc/mecabrc' \
    && make install

# Mecab-python 설치
RUN cd /tmp \
    && git clone https://bitbucket.org/eunjeon/mecab-python-0.996.git \
    && python -m pip install /tmp/mecab-python-0.996

# 작업 디렉토리 설정
WORKDIR /usr/src/app

# 필요한 파일 복사
COPY requirements.txt .

# 필요한 Python 패키지 설치
RUN pip install --no-cache-dir -r requirements.txt

# 나머지 소스 파일 복사
COPY . .

# 기본 명령어 설정 (예시)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
