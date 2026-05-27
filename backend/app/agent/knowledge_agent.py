"""Knowledge Agent — 知识库管理 Agent

职责:
- 维护 wiki 数据库的完整性和准确性
- 语义搜索 / 向量搜索 / 混合搜索
- 判断是否需要 create / update / delete
- 执行 CRUD 操作 + 同步到 Markdown + ChromaDB + Git
"""

from __future__ import annotations

from typing import Literal

from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from app.config import settings
from app.wiki import service

# LLM 用于决策
llm = ChatOpenAI(
    model=settings.ZHIPUAI_CHAT_MODEL,
    api_key=settings.ZHIPUAI_API_KEY,
    base_url=settings.ZHIPUAI_BASE_URL,
    temperature=0.3,
)


class KnowledgeDecision(BaseModel):
    """知识操作决策"""
    action: Literal["create", "update", "delete", "none"] = Field(description="操作类型")
    reason: str = Field(description="判断原因")
    path: str = Field(default="", description="知识条目路径，update/delete 时必填")
    title: str = Field(default="", description="条目标题，create 时必填")
    content: str = Field(default="", description="Markdown 内容，create/update 时必填")
    tags: list[str] | None = Field(default=None, description="标签列表")

    def to_dict(self) -> dict:
        return self.model_dump()


# PydanticOutputParser：自动生成格式指令 + 解析 + 校验
_parser = PydanticOutputParser(pydantic_object=KnowledgeDecision)

DECIDE_PROMPT = """你是一个知识库维护决策器。根据当前对话和现有知识库，判断是否需要创建、更新、删除知识条目。

## 当前对话
用户: {user_message}
AI: {ai_response}

## 现有知识库内容
{existing_knowledge}

## 判断规则
- create：对话中包含具有长期复用价值的新知识，且现有知识库没有相关条目。必须填写 title 和 content。
- update：对话对现有条目进行了补充、修正或结构化完善。必须填写 path 和 content。
- delete：用户明确要求删除某个知识条目。必须填写 path。
- none：普通问答、闲聊、重复已有内容、临时性问题、没有长期保存价值。

{format_instructions}
"""

# 检测模型是否支持 function calling（GLM 系列不支持，需要跳过）
_supports_function_calling = not any(
    keyword in settings.ZHIPUAI_CHAT_MODEL.lower()
    for keyword in ("glm", "zhipu")
)

# 仅在模型支持 function calling 时创建，否则为 None
structured_llm = llm.with_structured_output(KnowledgeDecision) if _supports_function_calling else None


def _get_related_knowledge(query: str) -> str:
    """获取与查询相关的现有知识"""
    results = service.search_pages(query)
    if not results:
        return "（无相关知识）"

    entries = []
    for r in results[:3]:
        try:
            page = service.get_page(r.path)
            preview = page.content[:500] + ("..." if len(page.content) > 500 else "")
            entries.append(f"### {r.title} ({r.path})\n{preview}")
        except Exception:
            entries.append(f"### {r.title} ({r.path})\n（读取失败）")
    return "\n\n".join(entries)


def _build_prompt(user_message: str, ai_response: str, existing_knowledge: str) -> str:
    """构建 prompt，注入 PydanticOutputParser 的格式指令"""
    return DECIDE_PROMPT.format(
        user_message=user_message,
        ai_response=ai_response,
        existing_knowledge=existing_knowledge,
        format_instructions=_parser.get_format_instructions(),
    )


async def decide_action(
    user_message: str,
    ai_response: str,
) -> KnowledgeDecision:
    """分析对话，决定是否需要对知识库进行操作

    策略 1: with_structured_output() — API 级别保证（仅支持 function calling 的模型）
    策略 2: PydanticOutputParser — prompt 指令 + 自动解析校验
    """
    existing_knowledge = _get_related_knowledge(user_message)
    prompt = _build_prompt(user_message, ai_response, existing_knowledge)

    # 策略 1: with_structured_output（模型支持 function calling 时生效）
    if structured_llm is not None:
        try:
            decision = await structured_llm.ainvoke([HumanMessage(content=prompt)])
            if isinstance(decision, KnowledgeDecision) and decision.action in ("create", "update", "delete", "none"):
                print(f"[Knowledge Agent] 结构化输出成功: action={decision.action}, reason={decision.reason}")
                return decision
            print(f"[Knowledge Agent] 结构化输出返回异常值，fallback 到 PydanticOutputParser")
        except Exception as e:
            print(f"[Knowledge Agent] 结构化输出失败，fallback: {e}")

    # 策略 2: PydanticOutputParser（自动提取 JSON + Pydantic 校验）
    try:
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        raw_text = (response.content or "").strip()
        print(f"[Knowledge Agent] Parser fallback ({len(raw_text)} chars): {raw_text[:300]}")

        if not raw_text:
            return KnowledgeDecision(action="none", reason="LLM 返回空内容")

        decision = _parser.parse(raw_text)
        print(f"[Knowledge Agent] PydanticOutputParser 成功: action={decision.action}, reason={decision.reason}")
        return decision
    except Exception as e:
        print(f"[Knowledge Agent] 决策失败: {e}")
        return KnowledgeDecision(action="none", reason=f"决策失败: {str(e)}")


async def search_knowledge(query: str, limit: int = 3) -> list[dict]:
    """搜索知识库

    Args:
        query: 搜索查询
        limit: 返回结果数量

    Returns:
        list[dict]: 搜索结果列表
    """
    results = service.search_pages(query)
    return [
        {
            "path": r.path,
            "title": r.title,
            "snippet": r.snippet,
            "score": r.score,
        }
        for r in results[:limit]
    ]


async def get_knowledge(path: str) -> dict | None:
    """获取知识条目

    Args:
        path: 知识条目路径

    Returns:
        dict: 知识条目内容，不存在返回 None
    """
    try:
        page = service.get_page(path)
        return {
            "path": page.path,
            "title": page.title,
            "content": page.content,
            "tags": page.tags,
            "created": page.created,
            "updated": page.updated,
        }
    except FileNotFoundError:
        return None


async def create_knowledge(
    title: str,
    content: str,
    category: str = "",
    tags: list[str] | None = None,
    source: str = "agent",
) -> dict:
    """创建知识条目

    Args:
        title: 条目标题
        content: Markdown 内容
        category: 分类路径
        tags: 标签列表
        source: 来源

    Returns:
        dict: 创建结果
    """
    from app.wiki.schemas import WikiPageCreate
    from app.wiki import git_service

    # 生成安全路径
    safe_title = title.strip().lower().replace(" ", "-")
    safe_title = "".join(c for c in safe_title if c.isalnum() or c in "-_一-龥")
    prefix = category.strip().rstrip("/") or "notes"
    path = f"{prefix}/{safe_title}.md"

    try:
        page = service.create_page(
            path,
            WikiPageCreate(
                title=title,
                content=content,
                tags=tags or [],
                source=source,
            ),
        )
        git_service.commit_changes(
            f"创建知识: {title}",
            files=[path],
        )
        return {
            "status": "ok",
            "action": "create",
            "path": path,
            "message": f"已创建: {page.title}",
        }
    except FileExistsError:
        return {
            "status": "error",
            "message": f"条目已存在: {path}",
        }


async def update_knowledge(
    path: str,
    title: str | None = None,
    content: str | None = None,
    tags: list[str] | None = None,
) -> dict:
    """更新知识条目

    Args:
        path: 知识条目路径
        title: 新标题（可选）
        content: 新内容（可选）
        tags: 新标签（可选）

    Returns:
        dict: 更新结果
    """
    from app.wiki.schemas import WikiPageUpdate
    from app.wiki import git_service

    try:
        page = service.update_page(
            path,
            WikiPageUpdate(
                title=title,
                content=content,
                tags=tags,
            ),
        )
        git_service.commit_changes(
            f"更新知识: {page.title}",
            files=[path],
        )
        return {
            "status": "ok",
            "action": "update",
            "path": path,
            "message": f"已更新: {page.title}",
        }
    except FileNotFoundError:
        return {
            "status": "error",
            "message": f"条目不存在: {path}",
        }


async def delete_knowledge(path: str) -> dict:
    """删除知识条目

    Args:
        path: 知识条目路径

    Returns:
        dict: 删除结果
    """
    from app.wiki import git_service

    try:
        service.delete_page(path)
        git_service.commit_changes(
            f"删除知识: {path}",
            files=[path],
        )
        return {
            "status": "ok",
            "action": "delete",
            "path": path,
            "message": f"已删除: {path}",
        }
    except FileNotFoundError:
        return {
            "status": "error",
            "message": f"条目不存在: {path}",
        }


async def get_knowledge_tree() -> dict:
    """获取知识库目录树

    Returns:
        dict: 目录树结构
    """
    tree = service.get_tree()
    return _tree_to_dict(tree)


def _tree_to_dict(node) -> dict:
    """递归转换目录树为字典"""
    result = {
        "name": node.name,
        "path": node.path,
        "is_dir": node.is_dir,
    }
    if node.children:
        result["children"] = [_tree_to_dict(child) for child in node.children]
    return result
