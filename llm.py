from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint
from langchain.chains import LLMChain
from dotenv import load_dotenv


load_dotenv()

def TypeBot(p1,p2):

    llm = HuggingFaceEndpoint(
        repo_id="HuggingFaceH4/zephyr-7b-beta",
        temperature =  0.7,
        provider="auto"
    )

    template = """
You are a expert in keybaord typing dynamics.

Here is a summary of a user cluster:
- User Typing: {level}
- Words per min: {wpm}

Give a natural-language behavioral profile of this group and things to improve for better typing result.
"""

    prompt = PromptTemplate.from_template(template)
    chain = prompt | llm

    result = chain.invoke(
    {"level":p1,
    "wpm":p2})
    return result


test = TypeBot("slow",37)
print(test)