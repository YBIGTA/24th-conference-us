import pandas as pd
import numpy as np
import faiss

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

'''매칭은 여기서 이루어지고, 아웃풋으로 매칭 결과인 User id와 추천 사유를 같이 제공'''

# 임베딩 + 취향 결합
def combine_features(embedding, preferences):
    return np.hstack((embedding, preferences))

# 데이터 전처리
combined_features = np.array([combine_features(row['text_embedding'], row['preferences']) for _, row in df.iterrows()])

# FAISS 인덱스 생성 및 훈련
d = combined_features.shape[1]  # 벡터 차원
index = faiss.IndexFlatL2(d)  # L2 거리 사용
index.add(combined_features)

# 유사도 검색 함수
def find_similar(user_embedding, k=2):
    distances, indices = index.search(np.array([user_embedding]), k)
    return distances, indices

user_id = 'user1'
user_embedding = combine_features(df[df['user_id'] == user_id]['text_embedding'].values[0],
                                  df[df['user_id'] == user_id]['preferences'].values[0])

distances, indices = find_similar(user_embedding)
recommended_indices = indices[0]
recommendations = df.iloc[recommended_indices]


# OpenAI API 키 설정
#llm = ChatOpenAI(api_key=openai_api_key, model='gpt-4')

user1 = '이동렬'
user1_txt = '나는 달리기가 좋고, 나무가 좋고, 초록색도 좋고...조용한 세상에서 책을 많이 읽고 시원한 바람 쐬고 와인 실컷 마시다가 죽고 싶다!'
user2 = '카리나'
user2_txt = '난 동렬이가 너무 좋아! 그리고.. 연예인 너무 지친다. 얼른 그만두고 떠나고 싶어. 어디론가 시원한 바람이 부는 곳으로, 조용한 곳으로..'

# 설명 템플릿 생성
template = PromptTemplate(
    input_variables=["user1", "user2", "user1_txt", "user2_txt", "similarity_score", "common_features"],
    template="{user1}님과 가장 잘 맞는 {user2}님은 {similarity_score}의 유사도로 추천되어요. 두분의 {user1_txt}와 {user2_txt}를 살펴보니, 이런 주제로 대화가 잘 통하실 것 같아요. {common_features}과 같은 공통된 특징도 있네요!"
)

# 템플릿을 사용한 LLMChain 생성
explain_chain = LLMChain(llm=llm, prompt=template)

# 공통 특징 추출 함수
def extract_common_features(user_embedding, recommendation_embedding):
    common_features = []
    for i in range(len(user_embedding)):
        if user_embedding[i] == recommendation_embedding[i]:
            common_features.append(f"feature {i+1}")
    return common_features

# 예시로 user1에 대한 설명 생성
user_embedding = [0.1, 0.2, 0.3]  # 실제 데이터로 대체
recommendation_embedding = [0.1, 0.2, 0.3]  # 실제 데이터로 대체
common_features = extract_common_features(user_embedding, recommendation_embedding)

explanation = explain_chain.run({
    "user1": user1,
    "user2": user2,
    "user1_txt": user1_txt,
    "user2_txt": user2_txt,
    "similarity_score": 0.85,  # 실제 유사도 점수로 대체
    "common_features": ', '.join(common_features)
})

