# Skill 索引（常驻层）

> 渐进式披露：此文件始终加载，仅包含 Skill 摘要（< 200 tokens）
> 当用户输入匹配触发条件时，自动加载完整 SKILL.md

## 可用 Skills

- `ppt-knowledge-extractor` — PPT 知识提取与总结
  触发词: ppt | pptx | 幻灯片 | 课件 | 提取PPT | 总结PPT | 课件解析
  描述: 从 PPT 文件提取文字、总结重点知识、搜索补充缺失内容、生成 Markdown 文档

## 加载策略

1. **常驻加载**: skill_index.md 始终注入，占用 < 200 tokens
2. **按需加载**: 匹配触发词后，加载对应 SKILL.md（500-2000 tokens）
3. **执行驻留**: Skill 执行期间完整驻留 Context
4. **任务完成**: 释放 Skill 定义，仅保留索引