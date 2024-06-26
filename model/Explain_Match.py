from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

openai_api_key = ""
llm = ChatOpenAI(api_key=openai_api_key, model_name="gpt-4")

def Explain_Rec(user1, user2):
    
    # 텍스트 읽기 및 user별 [이름, 답변] list 생성
    # user1 = [user1]    
    # for i in range(3):
    #     ans = open(f"./data/{user1}{i+1}.txt", "r")
    #     ans = ans.read()
    #     user1.append(ans)
    
    # user2 = [user2]
    # for i in range(3):
    #     ans = open(f"./data/{user2}{i+1}.txt", "r")
    #     ans = ans.read()
    #     user2.append(ans)
    
    
    # 프롬프트 
    rec_prompt = PromptTemplate(
    input_variables=["user1", "user2"],
    template=(f"""
    - role_system : 
    "당신은 인간 매칭 서비스의 대표입니다. 가치관과 관련한 사람들의 생각을 기반으로 두 명의 사람이 왜 매칭되었는지 설명하는 역할을 합니다."
    
    - role_assistant :
    Knowledge :	
    <가치관에 대한 3가지 질문> 
    1. 가장 영향을 많이 받은 사람/경험에 대해서 이야기해주세요. 
    2. 힘들고 어려운 난관을 어떻게 대하나요? 
    3. 10년 뒤 어떤 모습이고 싶나요? 
    
    - role_user :
    {user1[0]}, {user2[0]} 두 사람은 매칭되었습니다.	
    <가치관에 대한 3가지 질문>에 대한 두 사람의 답변이 주어집니다.	
    [{user1[0]}]	
    1. {user1[1]}
    2. {user1[2]}
    3. {user1[3]}
    [{user2[0]}]	
    1. {user2[1]}
    2. {user2[2]}
    3. {user2[3]}

    답변을 기반으로 두 사람이 어떤 가치관을 가지는지 정의하고, 두 사람의 생각과 가치관의 공통점을 기반으로 잘 맞는 이유를 설명하세요. 
    두 사람의 답변의 단어나 일부분을 참고해 '매칭된 이유'를 구체적으로 작성하세요. 절대 답변을 끼워맞추지 마세요. 
    """
    )
        )
    
    # 출력 
    rec_chain = LLMChain(llm=llm, prompt=rec_prompt)
    recommendation = rec_chain.run(
    user1=user1,
    user2=user2
    )
    
    return recommendation
         
