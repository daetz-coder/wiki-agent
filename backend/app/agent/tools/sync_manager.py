"""WikiSyncManager — Wiki 数据库同步管理器

确保 Markdown 文件、ChromaDB 向量索引、Git 版本 三者一致
"""

from __future__ import annotations

from pathlib import Path
from datetime import datetime

from app.config import settings
from app.wiki import service, git_service
from app.wiki.schemas import WikiPageCreate, WikiPageUpdate
from app.agent.tools.chunker import chunk_markdown
from app.agent.tools.bm25_index import get_bm25_index


class WikiSyncManager:
    """Wiki 数据库同步管理器"""

    def __init__(self):
        self._chroma_collection = None
        self._embedding_model = None

    @property
    def chroma_collection(self):
        """延迟初始化 ChromaDB collection"""
        if self._chroma_collection is None:
            try:
                import chromadb
                client = chromadb.PersistentClient(path=settings.CHROMA_DIR)
                self._chroma_collection = client.get_or_create_collection(
                    name="wiki_knowledge",
                    metadata={"hnsw:space": "cosine"},
                )
            except Exception as e:
                print(f"[WikiSync] ChromaDB 初始化失败: {e}")
                self._chroma_collection = None
        return self._chroma_collection

    @property
    def embedding_model(self):
        """延迟加载并缓存 Embedding 模型"""
        if self._embedding_model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL_PATH)
                print("[WikiSync] Embedding 模型已加载")
            except Exception as e:
                print(f"[WikiSync] Embedding 模型加载失败: {e}")
        return self._embedding_model

    def create(
        self,
        path: str,
        title: str,
        content: str,
        tags: list[str] | None = None,
        source: str = "agent",
    ) -> dict:
        """创建知识条目

        流程:
        1. 写入 Markdown 文件
        2. 生成 embedding 并存入 ChromaDB
        3. Git 提交

        Args:
            path: 知识条目路径
            title: 条目标题
            content: Markdown 内容
            tags: 标签列表
            source: 来源

        Returns:
            dict: 创建结果
        """
        try:
            # Step 1: 写入 Markdown 文件
            page = service.create_page(
                path,
                WikiPageCreate(
                    title=title,
                    content=content,
                    tags=tags or [],
                    source=source,
                ),
            )

            # Step 2: 更新 ChromaDB 向量索引
            self._sync_to_chroma(path, title, content, tags or [])

            # Step 3: Git 提交
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
        except Exception as e:
            return {
                "status": "error",
                "message": f"创建失败: {str(e)}",
            }

    def update(
        self,
        path: str,
        title: str | None = None,
        content: str | None = None,
        tags: list[str] | None = None,
    ) -> dict:
        """更新知识条目

        流程:
        1. 更新 Markdown 文件
        2. 更新 ChromaDB 向量索引
        3. Git 提交

        Args:
            path: 知识条目路径
            title: 新标题（可选）
            content: 新内容（可选）
            tags: 新标签（可选）

        Returns:
            dict: 更新结果
        """
        try:
            # Step 1: 更新 Markdown 文件
            page = service.update_page(
                path,
                WikiPageUpdate(
                    title=title,
                    content=content,
                    tags=tags,
                ),
            )

            # Step 2: 更新 ChromaDB 向量索引
            updated_content = content or page.content
            updated_title = title or page.title
            updated_tags = tags or page.tags
            self._sync_to_chroma(path, updated_title, updated_content, updated_tags)

            # Step 3: Git 提交
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
        except Exception as e:
            return {
                "status": "error",
                "message": f"更新失败: {str(e)}",
            }

    def delete(self, path: str) -> dict:
        """删除知识条目

        流程:
        1. 删除 Markdown 文件
        2. 删除 ChromaDB 向量
        3. Git 提交

        Args:
            path: 知识条目路径

        Returns:
            dict: 删除结果
        """
        try:
            # Step 1: 删除 Markdown 文件
            service.delete_page(path)

            # Step 2: 删除 ChromaDB 向量
            self._delete_from_chroma(path)

            # Step 3: Git 提交
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
        except Exception as e:
            return {
                "status": "error",
                "message": f"删除失败: {str(e)}",
            }

    def _sync_to_chroma(
        self,
        path: str,
        title: str,
        content: str,
        tags: list[str],
    ):
        """同步到 ChromaDB（带分块）

        Args:
            path: 知识条目路径
            title: 条目标题
            content: 条目内容
            tags: 标签列表
        """
        collection = self.chroma_collection
        if collection is None:
            print("[WikiSync] ChromaDB 不可用，跳过向量同步")
            return

        try:
            # 先删除该条目的所有旧分块
            self._delete_chunks_from_chroma(path)

            # 分块
            chunks = chunk_markdown(content, chunk_size=500, chunk_overlap=50)
            if not chunks:
                chunks = [content]

            # 为每个分块生成 embedding 并存储
            chunk_ids = []
            embeddings = []
            documents = []
            metadatas = []

            for i, chunk in enumerate(chunks):
                chunk_id = f"{path}#chunk{i}"
                chunk_ids.append(chunk_id)

                embedding = self._generate_embedding(f"{title}\n{chunk}")
                embeddings.append(embedding)

                documents.append(chunk)
                metadatas.append({
                    "path": path,
                    "title": title,
                    "tags": ",".join(tags),
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "updated_at": datetime.now().isoformat(),
                })

            # 批量添加
            collection.add(
                ids=chunk_ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
            )

            print(f"[WikiSync] ChromaDB 已同步: {path} ({len(chunks)} 块)")

            # 同步 BM25 索引
            bm25 = get_bm25_index()
            bm25.add_document(path, title, chunks)
            bm25.save()
            print(f"[WikiSync] BM25 已同步: {path} ({len(chunks)} 块)")
        except Exception as e:
            print(f"[WikiSync] ChromaDB 同步失败: {e}")

    def _delete_chunks_from_chroma(self, path: str):
        """删除指定条目的所有分块

        Args:
            path: 知识条目路径
        """
        collection = self.chroma_collection
        if collection is None:
            return

        try:
            # 查询该条目的所有分块
            results = collection.get(
                where={"path": path},
                include=[],
            )
            if results["ids"]:
                collection.delete(ids=results["ids"])
                print(f"[WikiSync] 已删除旧分块: {len(results['ids'])} 个")
        except Exception as e:
            # 如果 where 查询失败，尝试删除主记录
            try:
                collection.delete(ids=[path])
            except Exception:
                pass

    def _delete_from_chroma(self, path: str):
        """从 ChromaDB 和 BM25 删除（包括所有分块）

        Args:
            path: 知识条目路径
        """
        self._delete_chunks_from_chroma(path)

        # 同步删除 BM25 索引
        bm25 = get_bm25_index()
        bm25.remove_document(path)
        bm25.save()
        print(f"[WikiSync] BM25 已删除: {path}")

    def _generate_embedding(self, text: str) -> list[float]:
        """生成文本的向量嵌入

        Args:
            text: 输入文本

        Returns:
            list[float]: 向量嵌入
        """
        model = self.embedding_model
        if model is None:
            return [0.0] * 512
        try:
            embedding = model.encode(text)
            return embedding.tolist()
        except Exception as e:
            print(f"[WikiSync] Embedding 生成失败: {e}")
            return [0.0] * 512


# 全局实例
sync_manager = WikiSyncManager()
