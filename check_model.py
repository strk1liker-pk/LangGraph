"""本地配置检查；只有传入 --live 才向中转站发送付费测试请求。"""

import argparse

from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, MessagesState, StateGraph
from openai import APIError

from model_config import create_model


def run_checks(live: bool) -> int:
    model = create_model(max_tokens=256)
    print(f"模型: {model.model_name}")
    print(f"接口: {model.openai_api_base}")
    print("密钥: 已读取（不显示）")
    if not live:
        print("本地配置检查通过；使用 --live 测试实际调用（会消耗额度）。")
        return 0

    passed = 0

    def check(name, operation):
        nonlocal passed
        try:
            operation()
            passed += 1
            print(f"通过: {name}", flush=True)
        except Exception as exc:
            # 不打印第三方原始报错或请求体，避免泄露密钥及对话。
            status = getattr(exc, "status_code", None) if isinstance(exc, APIError) else None
            print(f"失败: {name} ({type(exc).__name__}, HTTP={status})", flush=True)

    def graph_call():
        builder = StateGraph(MessagesState)
        builder.add_node("chat", lambda state: {"messages": [model.invoke(state["messages"])]})
        builder.add_edge(START, "chat")
        builder.add_edge("chat", END)
        answer = builder.compile().invoke({"messages": [HumanMessage(content="Reply exactly OK.")]})
        assert answer["messages"][-1].text.strip() == "OK"

    def streaming_call():
        text = "".join(chunk.text for chunk in model.stream("Reply exactly OK."))
        assert text.strip() == "OK"

    def tool_roundtrip():
        @tool
        def add_numbers(a: int, b: int) -> int:
            """计算两个整数的和。"""
            return a + b

        history = [HumanMessage(content="Use add_numbers to calculate 2 + 3. After receiving the tool result, reply only with the number.")]
        with_tool = model.bind_tools([add_numbers], tool_choice="add_numbers")
        call = with_tool.invoke(history)
        assert len(call.tool_calls) == 1
        invocation = call.tool_calls[0]
        assert invocation["name"] == "add_numbers"
        result = add_numbers.invoke(invocation["args"])
        assert result == 5
        history.extend([call, ToolMessage(content=str(result), tool_call_id=invocation["id"])])
        answer = model.bind_tools([add_numbers], tool_choice="none").invoke(history)
        assert answer.text.strip() == "5"

    check("LangGraph 实际模型调用", graph_call)
    check("流式输出", streaming_call)
    check("工具调用及结果回传", tool_roundtrip)
    print(f"通过 {passed} 项，失败 {3 - passed} 项。")
    return 0 if passed == 3 else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="发送最多四次短模型请求，会消耗额度")
    args = parser.parse_args()
    try:
        raise SystemExit(run_checks(args.live))
    except ValueError:
        print("本地配置不完整或格式错误，请核对 .env 中的密钥、HTTPS 接口地址和模型名。")
        raise SystemExit(1)
