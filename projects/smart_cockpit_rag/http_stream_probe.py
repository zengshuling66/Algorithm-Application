import argparse #argparse 是标准库命令行参数解析器
import json
from pathlib import Path
from time import perf_counter
from urllib.request import Request, urlopen
from evaluation import load_cases

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--disconnect", action="store_true") #`action="store_true"表示带上--disconnect` 时为 True，否则为 False
    args = parser.parse_args()
    cases = load_cases(Path(__file__).with_name("eval_cases.json"))
    case = next(row for row in cases if row["id"] == "q_unlock_scope_001")
    body = json.dumps({"question": case["question"]}, ensure_ascii=False).encode("utf-8")
    #`Request` 描述请求
    request = Request("http://127.0.0.1:8001/qa/stream", data=body,
                      headers={"Content-Type": "application/json"}, method="POST")
    started, final, delta_count = perf_counter(), None, 0
    with urlopen(request, timeout=90) as response: #`urlopen` 实际发送
        print("HTTP:", response.status, flush=True)
        if response.headers.get_content_type() != "text/event-stream":
            raise ValueError("Expected text/event-stream")
        for raw in response: #逐行读取 SSE
            line = raw.decode("utf-8").rstrip("\r\n")
            if not line.startswith("data: "):
                continue
            event = json.loads(line[6:])
            print(f"{perf_counter() - started:.2f}s {event['type']}", flush=True)
            #时间表示客户端从发送请求到收到事件的耗时，包含检索和网络，不能直接当作纯模型耗时
            #正常模式必须收到 final；断开模式收到首个 delta 就退出，外层 with 随即关闭响应连接
            if event["type"] == "delta":
                delta_count += 1
                if args.disconnect:
                    break
            elif event["type"] == "error":
                raise RuntimeError(event["code"])
            elif event["type"] == "final":
                final = event["data"]
                print(json.dumps(final, ensure_ascii=False, indent=2), flush=True)
                break
    if args.disconnect and delta_count:
        print("status: client_closed")
    elif final is None:
        raise ValueError("Missing final event")
    else:
        print("status: completed", "delta_count:", delta_count)

if __name__ == "__main__":
    main()