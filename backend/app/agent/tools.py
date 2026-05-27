"""Agent 工具集 — 知识库操作"""

from __future__ import annotations

from langchain_core.tools import tool

from app.wiki import service


@tool
def wiki_search(query: str) -> str:
    """搜索知识库中的知识条目。传入关键词，返回匹配的条目标题、摘要和路径。

    Args:
        query: 搜索关键词
    """
    results = service.search_pages(query)
    if not results:
        return "未找到相关知识条目。"
    lines = []
    for r in results[:5]:
        lines.append(f"[{r.title}] 路径:{r.path}\n摘要: {r.snippet}")
    return "\n---\n".join(lines)


@tool
def wiki_read(path: str) -> str:
    """读取指定知识条目的完整内容。

    Args:
        path: 条目的文件路径，如 "programming/python/python-基础知识.md"
    """
    try:
        page = service.get_page(path)
        tags = ", ".join(page.tags) if page.tags else "无"
        return f"标题: {page.title}\n路径: {page.path}\n标签: {tags}\n来源: {page.source}\n\n{page.content}"
    except FileNotFoundError:
        return f"条目不存在: {path}"


@tool
def wiki_list(path: str = "") -> str:
    """列出知识库的目录结构。

    Args:
        path: 可选，分类路径，如 "programming/python"
    """
    try:
        tree = service.get_tree(path)
        lines = []
        for child in tree.children:
            prefix = "[目录]" if child.is_dir else "[文件]"
            lines.append(f"{prefix} {child.name}  路径:{child.path}")
        if not lines:
            return "该分类下暂无内容。"
        return "\n".join(lines)
    except Exception as e:
        return f"列出失败: {e}"


@tool
def wiki_create(title: str, content: str, category: str = "", tags: str = "") -> str:
    """创建一个新的知识条目。

    Args:
        title: 条目标题
        content: Markdown 格式的知识内容
        category: 知识分类路径，如 "programming/python"。留空则放入 "notes" 分类
        tags: 标签，多个用逗号分隔，如 "python,装饰器,设计模式"
    """
    from app.wiki.schemas import WikiPageCreate

    safe_title = title.strip().lower().replace(" ", "-")
    safe_title = "".join(c for c in safe_title if c.isalnum() or c in "-_一-龥")
    prefix = category.strip().rstrip("/") or "notes"
    path = f"{prefix}/{safe_title}.md"

    tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else []

    try:
        page = service.create_page(
            path,
            WikiPageCreate(title=title, content=content, tags=tag_list, source="agent"),
        )
        return f"已创建条目: {page.title}\n路径: {path}\n标签: {', '.join(tag_list) if tag_list else '无'}"
    except FileExistsError:
        return f"条目已存在: {path}，请使用 wiki_update 更新。"
    except Exception as e:
        return f"创建失败: {e}"


@tool
def wiki_update(path: str, content: str = "", title: str = "", tags: str = "") -> str:
    """更新已有的知识条目。

    Args:
        path: 条目的文件路径
        content: 新的 Markdown 内容（留空则不修改）
        title: 新标题（留空则不修改）
        tags: 新标签，逗号分隔（留空则不修改）
    """
    from app.wiki.schemas import WikiPageUpdate

    update_data = {}
    if title:
        update_data["title"] = title
    if content:
        update_data["content"] = content
    if tags:
        update_data["tags"] = [t.strip() for t in tags.split(",") if t.strip()]

    if not update_data:
        return "未提供任何修改内容。"

    try:
        page = service.update_page(path, WikiPageUpdate(**update_data))
        return f"已更新条目: {page.title}\n路径: {path}"
    except FileNotFoundError:
        return f"条目不存在: {path}，请使用 wiki_create 创建。"
    except Exception as e:
        return f"更新失败: {e}"


TOOLS = [wiki_search, wiki_read, wiki_list, wiki_create, wiki_update]
