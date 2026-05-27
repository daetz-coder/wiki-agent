"""Wiki API 路由"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from app.wiki import service, git_service
from app.wiki.schemas import (
    WikiPageCreate,
    WikiPageUpdate,
    WikiImportRequest,
    WikiNode,
    WikiPage,
    WikiCommit,
    WikiSearchResult,
)

router = APIRouter(prefix="/api/wiki", tags=["wiki"])


# ── 目录树 ──────────────────────────────────────────────────


@router.get("/tree", response_model=WikiNode)
def get_tree(path: str = ""):
    """获取目录树结构"""
    return service.get_tree(path)


# ── 搜索 ────────────────────────────────────────────────────


@router.get("/search", response_model=list[WikiSearchResult])
def search(q: str = Query(..., min_length=1)):
    """搜索知识条目"""
    return service.search_pages(q)


# ── 版本管理（必须在 CRUD 之前，避免 {path:path} 贪婪匹配）───


@router.get("/history", response_model=list[WikiCommit])
def get_global_history(limit: int = 50):
    """获取整个知识库的变更历史"""
    return git_service.get_history(limit=limit)


@router.get("/page/{path:path}/history", response_model=list[WikiCommit])
def get_history(path: str, limit: int = 20):
    """获取条目变更历史"""
    return git_service.get_history(rel_path=path, limit=limit)


@router.post("/page/{path:path}/rollback")
def rollback(path: str, commit_hash: str):
    """回滚条目到指定版本"""
    ok = git_service.rollback(path, commit_hash)
    if not ok:
        raise HTTPException(400, "回滚失败")
    return {"status": "ok", "message": f"已回滚到 {commit_hash}"}


# ── CRUD ────────────────────────────────────────────────────


@router.get("/page/{path:path}", response_model=WikiPage)
def get_page(path: str):
    """读取知识条目"""
    try:
        return service.get_page(path)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))


@router.post("/page/{path:path}", response_model=WikiPage, status_code=201)
def create_page(path: str, data: WikiPageCreate):
    """创建知识条目"""
    try:
        page = service.create_page(path, data)
        git_service.commit_changes(
            f"新建条目: {data.title}",
            files=[path],
        )
        return page
    except FileExistsError as e:
        raise HTTPException(409, str(e))


@router.put("/page/{path:path}", response_model=WikiPage)
def update_page(path: str, data: WikiPageUpdate):
    """更新知识条目"""
    try:
        page = service.update_page(path, data)
        git_service.commit_changes(
            f"更新条目: {page.title}",
            files=[path],
        )
        return page
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))


@router.delete("/page/{path:path}", status_code=204)
def delete_page(path: str):
    """删除知识条目"""
    try:
        service.delete_page(path)
        git_service.commit_changes(
            f"删除条目: {path}",
            files=[path],
        )
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))


# ── 导入 ────────────────────────────────────────────────────


@router.post("/import", response_model=WikiPage, status_code=201)
def import_markdown(data: WikiImportRequest):
    """导入 Markdown 内容"""
    try:
        if data.overwrite:
            page = service.update_page(
                data.path,
                WikiPageUpdate(content=data.content, source=data.source),
            )
        else:
            title = data.path.rsplit("/", 1)[-1].replace(".md", "")
            if data.content.startswith("# "):
                title = data.content.split("\n")[0][2:].strip()
            page = service.create_page(
                data.path,
                WikiPageCreate(
                    title=title, content=data.content, source=data.source
                ),
            )
        git_service.commit_changes(
            f"导入条目: {page.title} (来源: {data.source})",
            files=[data.path],
        )
        return page
    except FileExistsError:
        raise HTTPException(409, f"条目已存在: {data.path}，使用 overwrite=true 覆盖")
