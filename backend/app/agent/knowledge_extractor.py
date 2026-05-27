"""知识提取器 — 从对话中提取知识并判定操作"""

from __future__ import annotations

from typing import Literal

from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from app.config import settings
from app.wiki import service

# 使用较低温度确保输出稳定
llm = ChatOpenAI(
    model=settings.ZHIPUAI_CHAT_MODEL,
    api_key=settings.ZHIPUAI_API_KEY,
    base_url=settings.ZHIPUAI_BASE_URL,
    temperature=0.3,
)


class ExtractionResult(BaseModel):
    """知识提取结果"""
    action: Literal["create", "update", "none"] = Field(description="操作类型")
    reason: str = Field(description="判断原因")
    title: str = Field(default="", description="条目标题，action 为 create/update 时填写")
    category: str = Field(default="", description="分类路径如 programming/python，action 为 create 时填写")
    content: str = Field(default="", description="要保存的完整 Markdown 内容，action 为 create/update 时填写")
    target_path: str = Field(default="", description="更新目标路径如 programming/python/xxx.md，action 为 update 时填写")
    tags: list[str] | None = Field(default=None, description="标签列表")

    def to_dict(self) -> dict:
        return self.model_dump()


# PydanticOutputParser：自动生成格式指令 + 解析 + 校验
_parser = PydanticOutputParser(pydantic_object=ExtractionResult)

EXTRACT_PROMPT = """你是一个知识管理专家。分析以下对话，判断是否有值得保存到知识库的内容。

## 当前对话
用户: {user_message}
AI: {ai_response}

## 现有知识库目录结构
{wiki_tree}

## 现有相关条目（如有）
{existing_entries}

## 判断规则

1. **create** — 对话包含新知识，且知识库中没有相关条目。必须填写 title、category 和 content。
2. **update** — 对话补充或修正了现有条目。必须填写 target_path 和 content（更新后的完整内容）。
3. **none** — 对话没有知识保存价值（闲聊、简单问答、重复已有内容）。

注意：
- content 必须是完整的、结构化的 Markdown 内容
- 如果是 update，content 应该是更新后的完整内容，不是增量
- category 选择现有分类最合适的一个，或创建新的合理分类
- tags 要有意义，帮助后续检索

{format_instructions}
"""


def _get_wiki_tree_summary() -> str:
    """获取知识库目录结构的文本摘要"""
    try:
        tree = service.get_tree()
        return _format_tree(tree, 0)
    except Exception:
        return "（知识库为空）"


def _format_tree(node, depth: int) -> str:
    """递归格式化目录树"""
    lines = []
    if depth > 0:
        prefix = "  " * (depth - 1) + "├─ "
        label = "📁" if node.is_dir else "📄"
        lines.append(f"{prefix}{label} {node.name}")
    if node.children and depth < 3:
        for child in node.children:
            lines.append(_format_tree(child, depth + 1))
    return "\n".join(lines)


def _search_related_entries(query: str) -> str:
    """搜索与对话相关的现有条目"""
    results = service.search_pages(query)
    if not results:
        return "（无相关条目）"

    entries = []
    for r in results[:3]:
        try:
            page = service.get_page(r.path)
            preview = page.content[:500] + ("..." if len(page.content) > 500 else "")
            entries.append(f"### {r.title} ({r.path})\n{preview}")
        except Exception:
            entries.append(f"### {r.title} ({r.path})\n（读取失败）")
    return "\n\n".join(entries)


# 检测模型是否支持 function calling（GLM 系列不支持，需要跳过）
_supports_function_calling = not any(
    keyword in settings.ZHIPUAI_CHAT_MODEL.lower()
    for keyword in ("glm", "zhipu")
)

# 仅在模型支持 function calling 时创建，否则为 None
structured_llm = llm.with_structured_output(ExtractionResult) if _supports_function_calling else None


def _build_prompt(user_message: str, ai_response: str, wiki_tree: str, existing_entries: str) -> str:
    """构建 prompt，注入 PydanticOutputParser 的格式指令"""
    return EXTRACT_PROMPT.format(
        user_message=user_message,
        ai_response=ai_response,
        wiki_tree=wiki_tree,
        existing_entries=existing_entries,
        format_instructions=_parser.get_format_instructions(),
    )


async def extract_knowledge(
    user_message: str,
    ai_response: str,
) -> ExtractionResult:
    """从对话中提取知识，判定操作类型

    策略 1: with_structured_output() — API 级别保证（仅支持 function calling 的模型）
    策略 2: PydanticOutputParser — prompt 指令 + 自动解析校验
    """
    wiki_tree = _get_wiki_tree_summary()
    existing_entries = _search_related_entries(user_message)
    prompt = _build_prompt(user_message, ai_response, wiki_tree, existing_entries)

    # 策略 1: with_structured_output（模型支持 function calling 时生效）
    if structured_llm is not None:
        try:
            result = await structured_llm.ainvoke([HumanMessage(content=prompt)])
            if isinstance(result, ExtractionResult) and result.action in ("create", "update", "none"):
                print(f"[知识提取] 结构化输出成功: action={result.action}, reason={result.reason}, title={result.title}")
                return result
            print(f"[知识提取] 结构化输出返回异常值，fallback")
        except Exception as e:
            print(f"[知识提取] 结构化输出失败，fallback: {e}")

    # 策略 2: PydanticOutputParser（自动提取 JSON + Pydantic 校验）
    try:
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        raw_text = (response.content or "").strip()
        print(f"[知识提取] Parser fallback ({len(raw_text)} chars): {raw_text[:300]}")

        if not raw_text:
            return ExtractionResult(action="none", reason="LLM 返回空内容")

        result = _parser.parse(raw_text)
        print(f"[知识提取] PydanticOutputParser 成功: action={result.action}, reason={result.reason}, title={result.title}")
        return result
    except Exception as e:
        print(f"[知识提取] 提取失败: {e}")
        return ExtractionResult(action="none", reason=f"提取失败: {str(e)}")
