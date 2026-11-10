import os
from openai import OpenAI #OpenAI 是需要安装的 SDK 客户端类，SDK 帮我们封装请求和响应

#创建客户端
def build_client(): #读取服务端密钥和地址，返回客户端
    return OpenAI(
        api_key=os.environ["LLM_API_KEY"], base_url=os.environ["LLM_BASE_URL"],
        timeout=60.0, max_retries=0, #设置网络操作超时，暂时关闭 SDK 自动重试，让第一次接入的错误直接暴露
    )

#使用传入的客户端完成一次生成
#接收消息列表、现成客户端、请求模型名和 JSON 开关；执行一次生成，检查完成状态及空答案，返回原来的答案、实际模型名和用量
def generate_with_client(messages, client, model_name, json_mode=False):
    options = {"response_format": {"type": "json_object"}} if json_mode else {}
    response = client.chat.completions.create( #发起聊天生成请求
        model=model_name, messages=messages,
        max_tokens=1024, stream=False, #限制生成输出的 token 数，包括模型支持的思考输出和最终回答；stream=False表示等待完整响应，而不是逐段接收
        extra_body={"thinking": {"type": "disabled"}}, #关闭深度思考，extra_body 是 SDK 传递服务商扩展请求字段的位置
        **options, #如果 json_mode=True，就传入 response_format，要求模型输出 JSON 对象
    )
    choice = response.choices[0] #response.choices 是候选回复列表，[0] 取第一个候选
    if choice.finish_reason != "stop": #finish_reason 表示生成为什么结束；stop表示正常结束，本项目仅接受 stop，其余结束原因统一拒收
        raise ValueError(f"Incomplete generation: {choice.finish_reason}")
    answer = (choice.message.content or "").strip()
    #choice.message.content是最终回答文本；or "" 利用短路规则，在内容为空或为 None 时使用空字符串，避免直接调用 .strip() 出错
    if not answer:
        raise ValueError("Model returned an empty answer")
    return {
        "answer": answer, "model": response.model,
        "usage": response.usage.model_dump() if response.usage else None, #model_dump() 将用量对象转成字典
        #用量通常包含输入、输出和总 token，输入包含 Prompt 与资料，不只是用户问题。
    }

#负责创建、调用和关闭客户端
#未来服务启动 → 创建客户端 → 多次调用generate_with_client → 服务停机关闭
def generate_answer(messages, json_mode=False):
    model_name = os.environ["LLM_MODEL"]
    with build_client() as client: #退出 with 时负责关闭，正常返回或发生异常都会执行关闭
        return generate_with_client(messages, client, model_name, json_mode=json_mode)


# def generate_answer(messages, json_mode=False):
#     options = {"response_format": {"type": "json_object"}} if json_mode else {}
#     api_key = os.environ["LLM_API_KEY"]
#     base_url = os.environ["LLM_BASE_URL"]
#     model_name = os.environ["LLM_MODEL"]
#     with OpenAI( #创建客户端
#         api_key=api_key,
#         base_url=base_url,
#         timeout=60.0, #设置网络操作超时
#         max_retries=0, #暂时关闭 SDK 自动重试，让第一次接入的错误直接暴露
#     ) as client:
#         response = client.chat.completions.create( #发起聊天生成请求
#             model=model_name,
#             messages=messages,
#             max_tokens=1024, #限制生成输出的 token 数，包括模型支持的思考输出和最终回答
#             stream=False, #表示等待完整响应，而不是逐段接收
#             extra_body={"thinking": {"type": "disabled"}}, #关闭深度思考，extra_body 是 SDK 传递服务商扩展请求字段的位置
#             **options, #如果 json_mode=True，就传入 response_format，要求模型输出 JSON 对象
#         )
#     choice = response.choices[0] #response.choices 是候选回复列表，[0] 取第一个候选
#     if choice.finish_reason != "stop": #finish_reason 表示生成为什么结束
#         raise ValueError(f"Incomplete generation: {choice.finish_reason}")
#     answer = (choice.message.content or "").strip() #choice.message.content 是最终回答文本
#     #or "" 利用短路规则，在内容为空或为 None 时使用空字符串，避免直接调用 .strip() 出错
#     if not answer:
#         raise ValueError("Model returned an empty answer")
#     return {
#         "answer": answer,
#         "model": response.model,
#         "usage": response.usage.model_dump() if response.usage else None, #model_dump() 将用量对象转成字典
#         #用量通常包含输入、输出和总 token，输入包含 Prompt 与资料，不只是用户问题。
#     }