# MEMORY.md - 跨会话持久记忆

## 项目记录
- 2026-07-28: 初始化 Harness 架构，支持渐进式披露
- 2026-07-28: 创建 ppt-knowledge-extractor skill

## Skill 索引（常驻层）

| Skill 名称 | 触发条件 | 一句话描述 |
|---|---|---|
| ppt-knowledge-extractor | ppt/pptx/幻灯片/课件 | 提取PPT文字、总结知识、补充缺失内容 |

## 用户偏好
- 语言：中文
- 风格：通俗易懂，适合初学者
- 输出格式：Markdown

## 技术栈
- python-pptx: PPT 解析
- WebSearch: 知识补充
- Python: Harness 引擎