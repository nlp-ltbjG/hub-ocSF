#!/usr/bin/env python3
"""
Harness 入口文件

基于 Harness Engineering 设计的 AI Agent 框架
支持渐进式披露（Progressive Disclosure）架构

用法:
    python main.py                          # 启动交互模式
    python main.py --input "处理这个PPT"     # 单次执行
    python main.py --list-skills             # 列出可用 Skills
"""

import sys
import os
import argparse


def main():
    parser = argparse.ArgumentParser(
        description="Harness Engine - 渐进式披露架构",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python main.py                                    # 启动交互模式
  python main.py --input "分析 /path/to/file.pptx"  # 单次执行
  python main.py --list-skills                       # 查看可用技能
        """
    )

    parser.add_argument(
        "--input", "-i",
        type=str,
        help="用户输入（如 PPT 文件路径 + 指令）"
    )
    parser.add_argument(
        "--list-skills", "-l",
        action="store_true",
        help="列出所有可用 Skills"
    )
    parser.add_argument(
        "--status", "-s",
        action="store_true",
        help="显示 Harness 状态"
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        default=None,
        help="配置目录路径"
    )

    args = parser.parse_args()

    # 确定配置目录
    if args.config:
        config_dir = args.config
    else:
        config_dir = os.path.join(os.path.dirname(__file__), "config")

    if not os.path.exists(config_dir):
        print(f"错误: 配置目录不存在: {config_dir}")
        sys.exit(1)

    # 导入 Harness
    from harness import Harness

    # 初始化
    harness = Harness(config_dir)

    # 执行命令
    if args.list_skills:
        print("\n可用 Skills:")
        for skill in harness.disclosure.skill_index:
            triggers = ", ".join(skill.get("triggers", [])[:3])
            print(f"  • {skill['name']}")
            print(f"    描述: {skill.get('description', 'N/A')}")
            print(f"    触发: {triggers}")
            print()

    elif args.status:
        status = harness.get_status()
        print("\nHarness 状态:")
        for key, value in status.items():
            print(f"  {key}: {value}")

    elif args.input:
        result = harness.process_input(args.input)
        harness._display_result(result)

    else:
        # 交互模式
        harness.interactive_mode()


if __name__ == "__main__":
    main()