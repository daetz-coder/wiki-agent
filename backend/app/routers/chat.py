"""对话 API — Wiki Agent 统一服务"""

from __future__ import annotations

import json
from typing import AsyncGenerator

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from app.config import settings
from app.agent.graph import run_search, run_decide, resume_and_execute
from app.agent.tools import crud_tools
from app.session import store as session_store

router = APIRouter(prefix="/api/chat", tags=["chat"])

# Wiki Agent LLM
llm = ChatOpenAI(
    model=settings.ZHIPUAI_CHAT_MODEL,
    api_key=settings.ZHIPUAI_API_KEY,
    base_url=settings.ZHIPUAI_BASE_URL,
    temperature=0.7,
    streaming=True,
)

SYSTEM_PROMPT = """你是一个智能知识助手。请用中文回答。

## 你的能力
1. 用你的知识详细回答用户的各种问题
2. 搜索用户的个人知识库，如果有相关内容则引用并提供链接

## 回答要求
- 回答要详细、有结构、有价值
- 如果知识库有相关内容，在回答中引用并标注来源路径
- 如果知识库没有相关内容，直接用你的知识回答，不需要提及知识库
- 不要只说"已搜索"或"未找到"，要真正回答用户的问题
"""


class ChatRequest(BaseModel):
    session_id: str = "default"
    message: str


class SaveKnowledgeRequest(BaseModel):
    action: str  # "create" | "update" | "delete"
    title: str = ""
    category: str = ""
    content: str = ""
    target_path: str | None = None
    tags: list[str] = []


class ConfirmRequest(BaseModel):
    thread_id: str
    confirm: bool
    session_id: str | None = None


async def _ensure_session(session_id: str):
    """确保会话存在"""
    if not await session_store.session_exists(session_id):
        await session_store.create_session(session_id)


def _format_wiki_results(results: list[dict]) -> str:
    """格式化搜索结果"""
    lines = []
    for r in results[:3]:
        lines.append(f"- {r['title']} ({r['path']}): {r['snippet']}")
    return "\n".join(lines)


def _build_history(messages: list[dict]) -> list:
    """从数据库消息构建 LangChain 消息列表"""
    history = []
    for msg in messages:
        if msg["role"] == "user":
            history.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant" and msg["content"]:
            history.append(AIMessage(content=msg["content"]))
    return history


async def stream_response(session_id: str, user_message: str) -> AsyncGenerator[str, None]:
    """流式对话：Wiki Agent 统一服务"""
    # 确保会话存在
    await _ensure_session(session_id)

    # 获取历史消息
    session_data = await session_store.get_session(session_id)
    history = _build_history(session_data["messages"])
    history.append(HumanMessage(content=user_message))

    # 保存用户消息
    await session_store.add_message(session_id, "user", user_message)

    # 更新会话名称（使用第一条消息）
    if len(session_data["messages"]) == 0:
        name = user_message[:30] + ("..." if len(user_message) > 30 else "")
        await session_store.update_session_name(session_id, name)

    # Wiki Agent: 搜索知识库
    wiki_results, wiki_text = await run_search(user_message)

    # 构建带知识库上下文的消息
    if wiki_text:
        context_msg = f"[知识库搜索结果]\n{wiki_text}\n\n请结合以上知识库内容回答用户问题。如果知识库有相关内容，在回答中标注来源路径。"
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + history[:-1] + [
            SystemMessage(content=context_msg),
            HumanMessage(content=user_message),
        ]
        # 发送知识库搜索结果给前端
        yield f"data: {json.dumps({'type': 'wiki_results', 'results': wiki_text}, ensure_ascii=False)}\n\n"
    else:
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + history

    # Wiki Agent: 流式生成回复
    collected = ""
    try:
        async for chunk in llm.astream(messages):
            if chunk.content:
                collected += chunk.content
                yield f"data: {json.dumps({'type': 'content', 'text': chunk.content}, ensure_ascii=False)}\n\n"
    except Exception as e:
        yield f"data: {json.dumps({'type': 'error', 'message': str(e)}, ensure_ascii=False)}\n\n"

    # Wiki Agent: 分析对话，决定是否需要更新知识库（不自动执行）
    extraction_data = None
    print(f"[Wiki Agent] 用户消息: {user_message[:100]}...")
    print(f"[Wiki Agent] AI回复长度: {len(collected) if collected else 0}")

    if collected and len(collected) > 50:
        print("[Wiki Agent] 开始分析对话...")
        yield f"data: {json.dumps({'type': 'status', 'message': '正在分析对话内容...'}, ensure_ascii=False)}\n\n"
        try:
            # 只决策，不执行（等待用户确认）
            decide_result = await run_decide(user_message, collected)
            if decide_result:
                extraction_data = decide_result.get("decision")
                extraction_data["thread_id"] = decide_result.get("thread_id")
                print(f"[Wiki Agent] 决策完成（待确认）: {extraction_data}")
                yield f"data: {json.dumps({'type': 'extraction', 'data': extraction_data}, ensure_ascii=False)}\n\n"
            else:
                print("[Wiki Agent] 无需更新知识库")
        except Exception as e:
            print(f"[Wiki Agent] 分析异常: {e}")
            yield f"data: {json.dumps({'type': 'error', 'message': f'分析失败: {str(e)}'}, ensure_ascii=False)}\n\n"

    # 保存 AI 消息到数据库
    await session_store.add_message(
        session_id,
        "assistant",
        collected,
        wiki_results=wiki_text,
        extraction=extraction_data,
    )

    yield f"data: {json.dumps({'type': 'done'}, ensure_ascii=False)}\n\n"


@router.post("/stream")
async def chat_stream(req: ChatRequest):
    """SSE 流式对话"""
    return StreamingResponse(
        stream_response(req.session_id, req.message),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/message")
async def chat_message(req: ChatRequest):
    """非流式对话"""
    # 确保会话存在
    await _ensure_session(req.session_id)

    # 获取历史消息
    session_data = await session_store.get_session(req.session_id)
    history = _build_history(session_data["messages"])
    history.append(HumanMessage(content=req.message))

    # 保存用户消息
    await session_store.add_message(req.session_id, "user", req.message)

    # 更新会话名称
    if len(session_data["messages"]) == 0:
        name = req.message[:30] + ("..." if len(req.message) > 30 else "")
        await session_store.update_session_name(req.session_id, name)

    # Knowledge Agent: 搜索知识库
    wiki_results = search_tools.hybrid_search(req.message, limit=3)
    wiki_text = _format_wiki_results(wiki_results) if wiki_results else None

    if wiki_text:
        context_msg = f"[知识库搜索结果]\n{wiki_text}\n\n请结合以上知识库内容回答用户问题。"
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + history[:-1] + [
            SystemMessage(content=context_msg),
            HumanMessage(content=req.message),
        ]
    else:
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + history

    try:
        response = await llm.ainvoke(messages)

        # 保存 AI 回复
        await session_store.add_message(
            req.session_id,
            "assistant",
            response.content,
            wiki_results=wiki_text,
        )

        return {"content": response.content}
    except Exception as e:
        raise HTTPException(500, f"LLM 调用失败: {e}")


@router.post("/save-knowledge")
async def save_knowledge(req: SaveKnowledgeRequest):
    """保存提取的知识到知识库 — 使用 Knowledge Agent 的 CRUD 工具"""
    try:
        if req.action == "create":
            result = crud_tools.create_knowledge(
                title=req.title,
                content=req.content,
                category=req.category,
                tags=req.tags,
                source="chat-extraction",
            )
        elif req.action == "update":
            if not req.target_path:
                raise HTTPException(400, "更新操作需要 target_path")
            result = crud_tools.update_knowledge(
                path=req.target_path,
                title=req.title if req.title else None,
                content=req.content,
                tags=req.tags if req.tags else None,
            )
        elif req.action == "delete":
            if not req.target_path:
                raise HTTPException(400, "删除操作需要 target_path")
            result = crud_tools.delete_knowledge(req.target_path)
        else:
            raise HTTPException(400, f"不支持的操作: {req.action}")

        if result.get("status") == "error":
            error_msg = result.get("message", "未知错误")
            if "已存在" in error_msg:
                raise HTTPException(409, error_msg)
            elif "不存在" in error_msg:
                raise HTTPException(404, error_msg)
            else:
                raise HTTPException(500, error_msg)

        return result

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"操作失败: {str(e)}")


@router.post("/confirm")
async def confirm_knowledge(req: ConfirmRequest):
    """确认或取消知识库操作（Human-in-the-Loop）"""
    try:
        result = await resume_and_execute(req.thread_id, req.confirm)

        # 持久化 extraction 状态到数据库
        if req.session_id:
            status = "confirmed" if req.confirm else "rejected"
            await session_store.update_extraction_status(
                req.session_id, req.thread_id, status
            )

        return result
    except Exception as e:
        raise HTTPException(500, f"操作失败: {str(e)}")


@router.post("/sessions")
async def create_session(session_id: str = "default", name: str = "新对话"):
    """创建新会话"""
    await session_store.create_session(session_id, name)
    return {"id": session_id, "name": name}


@router.get("/sessions")
async def list_sessions():
    """列出所有会话"""
    sessions = await session_store.list_sessions()
    return {"sessions": sessions}


@router.get("/sessions/{session_id}")
async def get_session(session_id: str):
    """获取会话详情"""
    session = await session_store.get_session(session_id)
    if session is None:
        raise HTTPException(404, "会话不存在")
    return session


@router.delete("/sessions/{session_id}")
async def clear_session(session_id: str):
    """删除会话"""
    deleted = await session_store.delete_session(session_id)
    if not deleted:
        raise HTTPException(404, "会话不存在")
    return {"status": "ok"}
