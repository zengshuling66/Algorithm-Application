from langchain_core.prompts import ChatPromptTemplate #ChatPromptTemplate 负责聊天模板
from langchain_core.messages import convert_to_openai_messages #负责消息格式转换
from prompt_builder import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE, format_context

PROMPT_TEMPLATE = ChatPromptTemplate.from_messages([ #接收“角色＋模板”的元组列表
    ("system", SYSTEM_PROMPT),
    ("human", USER_PROMPT_TEMPLATE),
])

def build_langchain_messages(question, documents):
    question = question.strip()
    if not question:
        raise ValueError("question must not be empty")
    if not documents:
        raise ValueError("retrieval_results must not be empty")
    results = []
    for doc in documents:
        result = doc.metadata.copy()
        result["text"] = doc.page_content
        results.append(result)
    prompt_value = PROMPT_TEMPLATE.invoke({
        "question": question, "context": format_context(results),
        #format_context(results)得到证据字符串，再用 .invoke({...}) 填入两个同名占位符，返回 ChatPromptValue。
    })
    return convert_to_openai_messages(prompt_value.to_messages())
    #.to_messages() 取出消息对象，再转换为原 SDK 接受的 role/content 字典列表，其中 human 转成 user。