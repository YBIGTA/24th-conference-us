from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain


def Clean_Text(raw):
    openai_api_key = "sk-proj-HNMrLC5SL9il4okFNe2iT3BlbkFJUurTD1wsM4s4DI8HnMOX"
    llm = ChatOpenAI(api_key=openai_api_key, model_name="gpt-4")

    cleaning_prompt_template = PromptTemplate(
        input_variables=["text"],
        template="오타를 수정해줘. 정보를 수정하거나 추가하지 말고 맥락에 따라서 오타만 수정해줘. {text}"
    )
    clean_chain = LLMChain(llm=llm, prompt=cleaning_prompt_template)
    cleaned = clean_chain.run(text=raw)
    
    return cleaned