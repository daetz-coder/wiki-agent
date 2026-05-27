"""基于 GitPython 的知识库版本管理"""

from __future__ import annotations

from pathlib import Path

from git import Repo, InvalidGitRepositoryError

from app.config import settings
from app.wiki.schemas import WikiCommit

KNOWLEDGE_DIR = Path(settings.KNOWLEDGE_DIR)


def _get_repo() -> Repo | None:
    """获取 git repo，如果不存在则初始化"""
    if not settings.GIT_ENABLED:
        return None
    try:
        return Repo(KNOWLEDGE_DIR)
    except InvalidGitRepositoryError:
        return Repo.init(KNOWLEDGE_DIR)


def commit_changes(message: str, files: list[str] | None = None) -> str | None:
    """提交变更，返回 commit hash"""
    repo = _get_repo()
    if repo is None:
        return None

    if files:
        for f in files:
            repo.index.add([f])
    else:
        repo.index.add("*")

    # 检查是否有变更
    if not repo.index.diff("HEAD") and not repo.untracked_files:
        return None

    commit = repo.index.commit(message)
    return commit.hexsha[:8]


def get_history(rel_path: str | None = None, limit: int = 20) -> list[WikiCommit]:
    """获取变更历史"""
    repo = _get_repo()
    if repo is None:
        return []

    commits = []
    try:
        for commit in repo.iter_commits(paths=rel_path, max_count=limit):
            files = list(commit.stats.files.keys())
            commits.append(
                WikiCommit(
                    hash=commit.hexsha[:8],
                    message=commit.message.strip(),
                    date=commit.committed_datetime.isoformat(timespec="seconds"),
                    files=files,
                )
            )
    except Exception:
        pass
    return commits


def get_diff(rel_path: str, hash_a: str, hash_b: str = "HEAD") -> str:
    """获取两个版本之间的 diff"""
    repo = _get_repo()
    if repo is None:
        return ""

    try:
        a = repo.commit(hash_a)
        b = repo.commit(hash_b)
        diff = a.diff(b, paths=rel_path)
        if diff:
            return diff[0].diff.decode("utf-8", errors="replace")
    except Exception:
        pass
    return ""


def rollback(rel_path: str, commit_hash: str) -> bool:
    """回滚指定文件到指定版本"""
    repo = _get_repo()
    if repo is None:
        return False

    try:
        commit = repo.commit(commit_hash)
        blob = commit.tree / rel_path
        file_path = KNOWLEDGE_DIR / rel_path
        file_path.write_bytes(blob.data_stream.read())
        repo.index.add([rel_path])
        repo.index.commit(f"回滚 {rel_path} 到 {commit_hash[:8]}")
        return True
    except Exception:
        return False
