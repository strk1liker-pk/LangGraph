# LangGraph 学习项目

## 环境与模型

使用 Conda `langgraph` 环境（Python 3.13），依赖见 `requirements_full.txt`。
Notebook 内核选择 `Python (langgraph)`。

本地 `.env` 保存中转站密钥、接口和模型，配置项见 `.env.example`。
默认模型为 `gpt-5.6-terra`，使用 Responses 协议。
密钥仅从本项目 `.env` 读取，不读取 Codex 的账号配置；`.env` 已被 Git 忽略。
请求内容会经过中转站，费用及模型权限以中转站为准。

在项目根目录运行：

```powershell
conda activate langgraph
python check_model.py
# 可选：实际模型、流式输出和工具调用测试，最多四次短请求，会消耗额度。
python check_model.py --live
```

## 在练习中使用 Terra

课程原始 Notebook 和 HITL 示例仍保留 DeepSeek 配置，**不会自动切换**。
练习时，将 `ChatDeepSeek(...)` 的导入和初始化替换成：

```python
from model_config import create_model

model = create_model()
response = model.invoke("你好，用一句话介绍自己。")
print(response.text)
```

如果 Notebook 位于 `chapter01` 等子目录，先运行以下代码找到项目根目录：

```python
from pathlib import Path
import sys

cwd = Path.cwd().resolve()
root = next(p for p in (cwd, *cwd.parents) if (p / "model_config.py").is_file())
if str(root) not in sys.path:
    sys.path.insert(0, str(root))
```

Responses 返回的 `response.content` 可能是内容块列表；需要纯文字时用
`response.text`，流式输出时用 `chunk.text`。传递消息历史时保留完整消息对象，
不要只保存文字，以免丢失工具调用信息。

不要继续携带 DeepSeek 专用的 `extra_body={"thinking": ...}` 参数。
新入口默认关闭推理（`reasoning.effort="none"`），不主动请求服务端保存响应，
不自动重试；中转站的数据留存政策仍由其服务商决定。

官方模型参数参考：[GPT-5.6 Terra](https://developers.openai.com/api/docs/models/gpt-5.6-terra)。
