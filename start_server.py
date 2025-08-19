#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import subprocess
import time
import requests
from pathlib import Path


def check_dependencies():
    """检查依赖是否已安装"""
    try:
        import fastapi
        import uvicorn
        import jieba
        import sklearn
        print("✓ 依赖检查通过")
        return True
    except ImportError as e:
        print(f"✗ 缺少依赖: {e}")
        print("请运行: pip install -r requirements.txt")
        return False


def create_data_structure():
    """创建必要的数据目录结构"""
    data_dir = Path("data")
    images_dir = data_dir / "images"
    
    # 创建目录
    data_dir.mkdir(exist_ok=True)
    images_dir.mkdir(exist_ok=True)
    
    # 检查描述文件
    desc_file = data_dir / "descriptions.json"
    if not desc_file.exists():
        print("✓ 创建示例描述文件")
        # 描述文件已经在之前创建了
    
    print(f"✓ 数据目录结构已准备: {data_dir.absolute()}")
    return True


def start_server():
    """启动FastAPI服务器"""
    print("正在启动图片描述匹配API服务...")
    print("=" * 50)
    
    # 检查依赖
    if not check_dependencies():
        return False
    
    # 创建数据结构
    create_data_structure()
    
    # 启动服务
    try:
        print("\n启动服务器...")
        print("服务地址: http://localhost:8000")
        print("API文档: http://localhost:8000/docs")
        print("按 Ctrl+C 停止服务")
        print("-" * 50)
        
        # 使用uvicorn启动
        import uvicorn
        uvicorn.run(
            "main:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
            log_level="info"
        )
        
    except KeyboardInterrupt:
        print("\n服务已停止")
        return True
    except Exception as e:
        print(f"启动失败: {e}")
        return False


def test_server():
    """测试服务器是否正常运行"""
    print("测试服务器连接...")
    
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✓ 服务器运行正常")
            print(f"  状态: {data.get('status')}")
            print(f"  版本: {data.get('version')}")
            return True
        else:
            print(f"✗ 服务器响应异常: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("✗ 无法连接到服务器")
        print("请确保服务器已启动: python start_server.py")
        return False
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        return False


def show_usage():
    """显示使用说明"""
    print("图片描述匹配系统 - 启动脚本")
    print("=" * 40)
    print("使用方法:")
    print("  python start_server.py          # 启动服务器")
    print("  python start_server.py test     # 测试服务器")
    print("  python start_server.py demo     # 运行演示")
    print("  python start_server.py help     # 显示帮助")
    print()
    print("服务启动后可以:")
    print("  1. 访问 http://localhost:8000/docs 查看API文档")
    print("  2. 运行 python example_usage.py 测试API")
    print("  3. 运行 python example_usage.py demo 查看演示")


def run_demo():
    """运行演示"""
    print("运行API演示...")
    
    # 先测试服务器
    if not test_server():
        print("请先启动服务器: python start_server.py")
        return
    
    # 运行演示
    try:
        subprocess.run([sys.executable, "example_usage.py", "demo"])
    except Exception as e:
        print(f"演示运行失败: {e}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "test":
            test_server()
        elif command == "demo":
            run_demo()
        elif command == "help":
            show_usage()
        else:
            print(f"未知命令: {command}")
            show_usage()
    else:
        start_server()