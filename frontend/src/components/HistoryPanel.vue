<template>
  <div class="history-panel">
    <div class="history-header">
      <h2>变更流</h2>
      <span class="history-desc">知识库的所有变更记录</span>
    </div>

    <div v-if="loading" class="history-loading">加载中...</div>

    <div v-else-if="commits.length === 0" class="history-empty">
      暂无变更记录
    </div>

    <div v-else class="history-list">
      <div
        v-for="commit in commits"
        :key="commit.hash"
        class="commit-item"
      >
        <div class="commit-dot"></div>
        <div class="commit-content">
          <div class="commit-message">{{ commit.message }}</div>
          <div class="commit-meta">
            <span class="commit-hash">{{ commit.hash }}</span>
            <span class="commit-time">{{ formatTime(commit.date) }}</span>
            <span class="commit-files">
              {{ commit.files.join(", ") }}
            </span>
          </div>
          <div class="commit-actions">
            <button
              v-for="file in commit.files"
              :key="file"
              class="btn-file"
              @click="$emit('select', file)"
            >
              {{ file }}
            </button>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted } from "vue";
import { wikiApi } from "../api/index.js";

defineEmits(["select"]);

const commits = ref([]);
const loading = ref(true);

async function loadHistory() {
  loading.value = true;
  try {
    // 获取整个知识库的 git 历史
    commits.value = await wikiApi.getGlobalHistory();
  } catch (e) {
    console.error("加载历史失败:", e);
    commits.value = [];
  } finally {
    loading.value = false;
  }
}

function formatTime(dateStr) {
  if (!dateStr) return "";
  const d = new Date(dateStr);
  const now = new Date();
  const diff = now - d;
  if (diff < 60000) return "刚刚";
  if (diff < 3600000) return `${Math.floor(diff / 60000)} 分钟前`;
  if (diff < 86400000) return `${Math.floor(diff / 3600000)} 小时前`;
  if (diff < 604800000) return `${Math.floor(diff / 86400000)} 天前`;
  return d.toLocaleDateString("zh-CN");
}

onMounted(loadHistory);
</script>

<style scoped>
.history-panel {
  flex: 1;
  overflow-y: auto;
  padding: 24px 32px;
}

.history-header {
  margin-bottom: 24px;
}

.history-header h2 {
  font-size: 20px;
  font-weight: 600;
  margin-bottom: 4px;
}

.history-desc {
  font-size: 13px;
  color: #888;
}

.history-loading,
.history-empty {
  text-align: center;
  padding: 40px;
  color: #999;
}

.history-list {
  position: relative;
  padding-left: 20px;
}

/* 时间线竖线 */
.history-list::before {
  content: "";
  position: absolute;
  left: 7px;
  top: 0;
  bottom: 0;
  width: 2px;
  background: #e8e8e8;
}

.commit-item {
  position: relative;
  padding: 0 0 24px 16px;
}

.commit-dot {
  position: absolute;
  left: -17px;
  top: 6px;
  width: 10px;
  height: 10px;
  border-radius: 50%;
  background: #4a90d9;
  border: 2px solid #fff;
  box-shadow: 0 0 0 2px #e8e8e8;
}

.commit-content {
  background: #fff;
  border: 1px solid #e8e8e8;
  border-radius: 8px;
  padding: 14px 18px;
}

.commit-message {
  font-size: 14px;
  font-weight: 500;
  color: #1a1a1a;
  margin-bottom: 8px;
}

.commit-meta {
  display: flex;
  gap: 12px;
  font-size: 12px;
  color: #999;
  margin-bottom: 10px;
}

.commit-hash {
  font-family: monospace;
  background: #f5f5f5;
  padding: 1px 6px;
  border-radius: 3px;
  color: #666;
}

.commit-files {
  color: #888;
}

.commit-actions {
  display: flex;
  gap: 6px;
  flex-wrap: wrap;
}

.btn-file {
  padding: 2px 10px;
  background: #f0f7ff;
  border: 1px solid #d0e3f7;
  border-radius: 4px;
  cursor: pointer;
  font-size: 11px;
  color: #4a90d9;
  transition: all 0.15s;
}

.btn-file:hover {
  background: #e0efff;
  border-color: #4a90d9;
}
</style>
