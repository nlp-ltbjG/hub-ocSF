#!/usr/bin/env python3
"""测试 peoples_daily 数据集处理是否正确"""
import sys
sys.path.insert(0, 'src')

from dataset import build_label_schema, load_records, PeoplesDailyDataset
from transformers import BertTokenizer

def main():
    # 构建标签体系
    labels, label2id, id2label = build_label_schema()
    print(f"标签体系 ({len(labels)}个标签):")
    for i, lbl in enumerate(labels):
        print(f"  {i}: {lbl}")
    print()

    # 加载数据
    train_records = load_records("train")
    val_records = load_records("validation")
    test_records = load_records("test")
    print(f"数据集规模: 训练={len(train_records)}, 验证={len(val_records)}, 测试={len(test_records)}")
    print()

    # 查看第一条训练数据
    print("第一条训练数据:")
    print(f"  tokens: {train_records[0]['tokens']}")
    print(f"  ner_tags: {train_records[0]['ner_tags']}")
    print()

    # 统计实体分布
    entity_counts = {}
    for record in train_records:
        for tag in record['ner_tags']:
            if tag != 'O':
                etype = tag.split('-')[1] if '-' in tag else tag
                entity_counts[etype] = entity_counts.get(etype, 0) + 1
    
    print("训练集实体分布:")
    for etype, cnt in sorted(entity_counts.items(), key=lambda x: -x[1]):
        print(f"  {etype}: {cnt} 个")

if __name__ == "__main__":
    main()
