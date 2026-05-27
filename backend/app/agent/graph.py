"""Wiki Agent — 统一的知识助手 Agent

职责:
- 搜索知识库
- 生成回复（结合知识库内容）
- 决策是否需要更新知识库
- 执行 CRUD 操作（需用户确认后执行，Human-in-the-Loop）
"""

from __future__ import annotations

import uuid
from typing import Literal, TypedDict

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import interrupt, Command

from app.agent import knowledge_agent
from app.agent.tools import search_tools, crud_tools


# ── 状态定义 ──────────────────────────────────────────────────

class WikiState(TypedDict):
    """Wiki Agent 共享状态"""
    user_message: str
    wiki_results: list[dict]
    wiki_text: str | None
    ai_response: str
    decision: dict | None
    action_result: dict | None
    stage: str  # search → respond → decide → execute → done


# ── Checkpoint 存储（持久化到 SQLite，重启不丢失）────────────────

import os
import aiosqlite
from app.config import settings

_CHECKPOINT_DB = os.path.join(os.path.dirname(settings.DB_PATH), "checkpoints.db")
os.makedirs(os.path.dirname(_CHECKPOINT_DB), exist_ok=True)


# ── 节点函数 ──────────────────────────────────────────────────

async def search(state: WikiState) -> WikiState:
    """Wiki Agent: 搜索知识库"""
    print("[Wiki Agent] 搜索知识库...")
    user_message = state["user_message"]

    # 混合搜索
    results = search_tools.hybrid_search(user_message, limit=3)

    # 格式化结果
    wiki_text = None
    if results:
        lines = []
        for r in results[:3]:
            lines.append(f"- {r['title']} ({r['path']}): {r['snippet']}")
        wiki_text = "\n".join(lines)

    return {
        **state,
        "wiki_results": results,
        "wiki_text": wiki_text,
        "stage": "search",
    }


async def respond(state: WikiState) -> WikiState:
    """Wiki Agent: 生成回复（占位，实际回复在流式处理中生成）"""
    print("[Wiki Agent] 生成回复...")
    return {
        **state,
        "stage": "respond",
    }


async def decide(state: WikiState) -> WikiState:
    """Wiki Agent: 分析对话，决定是否需要更新知识库"""
    print("[Wiki Agent] 分析对话...")
    user_message = state["user_message"]
    ai_response = state.get("ai_response", "")

    if not ai_response or len(ai_response) < 50:
        return {
            **state,
            "decision": {"action": "none", "reason": "回复太短"},
            "stage": "decide",
        }

    # 调用 Knowledge Agent 决策
    decision = await knowledge_agent.decide_action(user_message, ai_response)
    decision_dict = decision.to_dict()

    # title 为空时从 path 推导
    if not decision_dict.get("title") and decision_dict.get("path"):
        stem = decision_dict["path"].replace(".md", "").split("/")[-1]
        decision_dict["title"] = stem

    return {
        **state,
        "decision": decision_dict,
        "stage": "decide",
    }


async def execute(state: WikiState) -> WikiState:
    """Wiki Agent: 执行知识库操作（interrupt() 暂停等待用户确认）"""
    # interrupt() 返回 Command(resume=...) 传入的值（True/False）
    user_confirmed = interrupt({})

    if not user_confirmed:
        print("[Wiki Agent] 用户取消操作")
        return {
            **state,
            "action_result": {"status": "cancelled", "message": "用户取消"},
            "stage": "execute",
        }

    decision = state.get("decision")
    if not decision or decision.get("action") == "none":
        return {
            **state,
            "action_result": None,
            "stage": "execute",
        }

    action = decision.get("action")
    print(f"[Wiki Agent] 用户确认，执行操作: {action}")

    result = None
    if action == "create":
        result = crud_tools.create_knowledge(
            title=decision.get("title", ""),
            content=decision.get("content", ""),
            category=decision.get("category", ""),
            tags=decision.get("tags", []),
        )
    elif action == "update":
        result = crud_tools.update_knowledge(
            path=decision.get("path", ""),
            content=decision.get("content"),
            tags=decision.get("tags"),
        )
    elif action == "delete":
        result = crud_tools.delete_knowledge(decision.get("path", ""))

    print(f"[Wiki Agent] 执行结果: {result}")
    return {
        **state,
        "action_result": result,
        "stage": "execute",
    }


# ── 条件路由 ──────────────────────────────────────────────────

def should_decide(state: WikiState) -> Literal["decide", "end"]:
    """决定是否需要分析对话"""
    ai_response = state.get("ai_response", "")
    if ai_response and len(ai_response) > 50:
        return "decide"
    return "end"


def should_execute(state: WikiState) -> Literal["execute", "end"]:
    """决定是否需要执行操作"""
    decision = state.get("decision")
    if decision and decision.get("action") != "none":
        return "execute"
    return "end"


# ── 构建图 ──────────────────────────────────────────────────

def create_wiki_graph(checkpointer):
    """创建 Wiki Agent 编排图

    流程:
    1. search: 搜索知识库
    2. respond: 生成回复（外部处理）
    3. decide: 分析对话，决定操作
    4. [interrupt] → 等待用户确认
    5. execute: 执行 CRUD 操作（用户确认后才执行）

    Returns:
        CompiledStateGraph: 编译后的图（带 checkpoint + interrupt）
    """
    graph = StateGraph(WikiState)

    # 添加节点
    graph.add_node("search", search)
    graph.add_node("respond", respond)
    graph.add_node("decide", decide)
    graph.add_node("execute", execute)

    # 设置入口
    graph.set_entry_point("search")

    # 添加边
    graph.add_edge("search", "respond")
    graph.add_conditional_edges(
        "respond",
        should_decide,
        {
            "decide": "decide",
            "end": END,
        },
    )
    graph.add_conditional_edges(
        "decide",
        should_execute,
        {
            "execute": "execute",
            "end": END,
        },
    )
    graph.add_edge("execute", END)

    # 编译：启用 checkpoint（interrupt 在 execute 节点内部调用）
    return graph.compile(checkpointer=checkpointer)


# ── 便捷函数 ──────────────────────────────────────────────────

# 全局图实例
_wiki_graph = None


async def get_wiki_graph():
    """获取 Wiki Agent 图实例（延迟异步初始化）"""
    global _wiki_graph
    if _wiki_graph is None:
        conn = await aiosqlite.connect(_CHECKPOINT_DB)
        checkpointer = AsyncSqliteSaver(conn=conn)
        _wiki_graph = create_wiki_graph(checkpointer)
    return _wiki_graph


async def run_search(user_message: str) -> tuple[list[dict], str | None]:
    """运行搜索阶段

    Args:
        user_message: 用户消息

    Returns:
        tuple: (搜索结果列表, 格式化文本)
    """
    results = search_tools.hybrid_search(user_message, limit=3)

    wiki_text = None
    if results:
        lines = []
        for r in results[:3]:
            lines.append(f"- {r['title']} ({r['path']}): {r['snippet']}")
        wiki_text = "\n".join(lines)

    return results, wiki_text


async def run_decide(
    user_message: str,
    ai_response: str,
) -> dict | None:
    """运行决策阶段（不执行操作，等待用户确认）

    Args:
        user_message: 用户消息
        ai_response: AI 回复

    Returns:
        dict: 决策结果（含 thread_id 用于后续 resume），action=none 时返回 None
    """
    if not ai_response or len(ai_response) < 50:
        return None

    graph = await get_wiki_graph()
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    # 运行图到 execute 节点内部的 interrupt() 暂停
    result = await graph.ainvoke(
        {
            "user_message": user_message,
            "ai_response": ai_response,
        },
        config,
    )

    decision = result.get("decision")
    if not decision or decision.get("action") == "none":
        print("[Wiki Agent] 无需更新知识库")
        return None

    print(f"[Wiki Agent] 决策: action={decision.get('action')}, reason={decision.get('reason')}")
    return {
        "thread_id": thread_id,
        "decision": decision,
    }


async def resume_and_execute(
    thread_id: str,
    confirm: bool,
) -> dict:
    """从 checkpoint 恢复图，执行或取消操作

    Args:
        thread_id: 之前 run_decide 返回的 thread_id
        confirm: 用户是否确认执行

    Returns:
        dict: 执行结果
    """
    graph = await get_wiki_graph()
    config = {"configurable": {"thread_id": thread_id}}

    # 恢复图，Command(resume=...) 传入 interrupt 返回值
    result = await graph.ainvoke(Command(resume=confirm), config)

    action_result = result.get("action_result")
    decision = result.get("decision", {})

    if action_result and action_result.get("status") == "cancelled":
        print(f"[Wiki Agent] 用户取消: {decision.get('action')}")
        return {
            "status": "cancelled",
            "message": "用户取消操作",
            "decision": decision,
        }

    print(f"[Wiki Agent] 执行完成: {action_result}")
    return {
        "status": "ok",
        "decision": decision,
        "result": action_result,
    }
