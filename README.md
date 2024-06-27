# 최적의 대화 상대 제공, Us🌏 입니다.

<div align="center">
<h3>24-1 YBIGTA 컨퍼런스</h3>

<em> 최적의 대화 상대 제공, Us🌏입니다. </em>

</div>

## 목차
- [문제 정의](#문제-정의)
- [선행 연구](#선행-연구)
- [세부 목표](#세부-목표)
- [접근 방법](#접근-방법)
- [결과 및 주요 기능](#결과-및-주요-기능)
- [팀 구성](#팀-구성)

## 💊 문제 정의
*💡 좋은 대화를 제공하기 위해 매칭 기능과 설명을 제공한다*
*💡 우리에겐 쉼이 되어주는 커뮤니티는 얼마 없다. 자주 보는 사람들에게 마음을 터놓고 이야기를 하기 쉽지 않고, 판단 당할 것 같은, 가까우니까 더 하기 어려운 말들이 있다. 이를 언어적/비언어적 요소로 분리해 매칭 기능을 구현했다.*

## 📃 선행 연구

*LLM을 이용한 추천 시스템의 큰 두갈래*
  1. LLM을 이용한 Recommendation System
  2. LLM으로 설명을 제공하는 Recommendation System

## ⛳️ 세부 목표

*단순한 유사도 접근보다, 표면적이지 않은 요소들, 본질에 접근하려고 노력했다.*
  1. 대화에 중요한 요소 🗣️
     - 발화에서 나타난 텐션 : 텐션이 비슷한 사람끼리 대화가 잘 맞을거라는 가정.
     - 발화의 높낮이 : 주파수와 진폭 외의 피처들을 이용해 유저의 특징을 포착하려고 시도함.
  2. 추천된 이유의 설명 🗒️
     - Cold Start의 문제점은 데이터의 양이 부족해 딥러닝 등 깊은 단계의 추천을 보여주기 어렵다는 것에 있다.
     - 따라서 LLM을 이용해 추천된 사용자 간의 사유 설명을 시도한다.  

## 🙌 접근 방법

1. **🗂️ 태스크** *(세부 목표를 달성하기 위한 구체적인 태스크)*
   - 프론트팀 
     - Figma + React 구현
   - 모델팀
     - Rule Based Recommendation
       - 발화에서 나타난 텐션 : 텐션이 비슷한 사람끼리 대화가 잘 맞을거라는 가정.
       - 발화의 높낮이 : 주파수와 진폭 외의 피처들을 이용해 유저의 특징을 포착하려고 시도함.
       - 텍스트 피처 : 녹음된 유저의 대화를 센텐스 임베딩 값을 계산하여 잘 맞는 사람을 서칭함.
       - 포스 태깅 : 유저의 말투를 판단하여 유저간 매칭에 가중치 추가.
       - 취향 유사도 : 선택된 유저들의 취향을 통해 매칭 알고리즘에 추가.
     - Explainable Recommendation
       - 추천 사유를 LLM을 이용해 설명
       - GPT-4o 모델을 사용
     - Text Cleaning
       - 좋지 않은 품질의 텍스트에 대해서 LLM을 통해 문맥정리 및 단어 정리
       


3. **🗄️ 데이터셋** *(사용한 데이터셋, API 등)*
    - 직접 수집
        - 학회원님들께 감사드립니다! 

4. **🗃️ 모델링/아키텍쳐 등** *(프로젝트 특성 및 목표에 따라)*
    - 모델
      <img width="1089" alt="Screenshot 2024-06-28 at 1 16 48 AM" src="https://github.com/Uarth/Us/assets/138357845/211e6969-dd8b-4758-aac1-8791ad837f43">
    - 서비스 아키텍쳐
      ![image](https://github.com/Uarth/Us/assets/138357845/1523a789-6e5c-4145-a309-bf951b8b8712)


## 🖥️ 결과 및 주요 기능

*📍음성을 녹음한 후, 답변의 내용과 보이스 특성을 기반으로 사람을 매칭*
### 유저 화면 예시 
<img width="855" alt="image" src="https://github.com/Uarth/Us/assets/87052350/b5f94b1f-0764-48c5-85d3-d0e0df988e91">
<img width="859" alt="image" src="https://github.com/Uarth/Us/assets/87052350/39e7d613-c37b-4997-9b7d-858a4e8e0693">

## 👥 팀 구성

|이름|팀|역할|
|-|-|-|
|(🦩이동렬)|DS|(이동렬)|
|(🐼김대솔)|DS|(김대솔)|
|(🐯임채림)|DE|(임채림)|
|(🦝목종원)|DE|(목종원)|
|(🐲김인영)|DA|()|
|(🦄정지원)|DA|()|
|(🦊김지훈)|DS|(김지훈)|
