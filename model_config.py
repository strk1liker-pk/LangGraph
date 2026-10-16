"""课程练习共用的模型入口；仅从项目 .env 读取密钥。"""

from pathlib import Path
from urllib.parse import urlsplit

from dotenv import dotenv_values
from langchain_openai import ChatOpenAI


ENV_PATH = Path(__file__).resolve().parent / ".env"


def create_model(*, max_tokens: int | None = None) -> ChatOpenAI:
    # 固定项目路径，避免 Notebook 工作目录变化或系统旧密钥影响配置。
    config = dotenv_values(ENV_PATH, interpolate=False)
    key = (config.get("OPENAI_API_KEY") or "").strip()
    if not key or key == "填写中转站新密钥":
        raise ValueError("请在项目 .env 中填写 OPENAI_API_KEY，不要把密钥写入代码。")
    base_url = (config.get("OPENAI_BASE_URL") or "https://shuxin79.top/v1").strip().rstrip("/")
    address = urlsplit(base_url)
    if address.scheme != "https" or not address.hostname or address.username or address.password or address.query or address.fragment:
        raise ValueError("OPENAI_BASE_URL 必须为不含凭据、查询参数或片段的 HTTPS 地址。")
    model = (config.get("OPENAI_MODEL") or "gpt-5.6-terra").strip()
    if not model:
        raise ValueError("OPENAI_MODEL 不能为空。")

    return ChatOpenAI(
        api_key=key,
        base_url=base_url,
        model=model,
        use_responses_api=True,
        # 保留 Responses 的内容块及工具信息；纯文字用 response.text 获取。
        output_version="responses/v1",
        reasoning={"effort": "none"},
        store=False,
        timeout=45,
        # 避免测试失败时自动重试产生额外费用。
        max_retries=0,
        max_tokens=max_tokens,
    )
