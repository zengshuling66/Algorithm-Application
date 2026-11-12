import json
import os
from contextlib import closing
from generation import build_client, stream_with_client

def main():
    messages = [{"role": "user", "content": '请只输出 JSON 对象：{"answer":"API_OK"}'}]
    final, delta_count = None, 0 #`final` 保存完成事件，`delta_count` 统计有正文的片段数

#closing来自标准库：退出时调用生成器的 .close()，让它内部的响应流退出；外层 with build_client()再关闭客户端
    with build_client() as client:
        with closing(stream_with_client(
            messages, client, os.environ["LLM_MODEL"], json_mode=True,
        )) as events:
            for event in events:
                if event["type"] == "delta":
                    delta_count += 1
                    print(event["text"], end="", flush=True)
                    #每次收到 delta 就打印，`end="" 不额外换行，flush=True` 要求 Python 立即刷新输出，而不是等完整答案拼好再打印
                elif event["type"] == "generation_done": #收到 generation_done 时先保存
                    final = event

    #循环结束后，必须同时满足确实收到完成事件、完整文本能解析成预期 JSON，才打印 status: completed
    if final is None:
        raise ValueError("Missing generation_done event")
    if json.loads(final["answer"]) != {"answer": "API_OK"}:
        raise ValueError("Unexpected probe answer")

    print("\nstatus: completed")
    print("text chunks:", delta_count)
    print("model:", final["model"])
    print("usage:", final["usage"])

if __name__ == "__main__":
    main()