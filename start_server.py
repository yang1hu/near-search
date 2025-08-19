#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import uvicorn
from pathlib import Path

def check_dependencies():
    """检查依赖包是否安装"""
    try:
        import fastapi
        import jieba
        import sklearn
        import numpy
        print("✓ 所有依赖包已安装")
        return True
    except ImportError as e:
        print(f"✗ 缺少依赖包: {e}")
        print("请运行: pip install -r requirements.txt")
        return False

def setup_data_directory():
    """设置数据目录"""
    data_dir = Path("data")
    images_dir = data_dir / "images"
    
    # 创建目录
    data_dir.mkdir(exist_ok=True)
    images_dir.mkdir(exist_ok=True)
    
    # 检查描述文件
    desc_file = data_dir / "descriptions.json"
    if not desc_file.exists():
        print("✓ 创建默认描述文件")
    else:
        print("✓ 描述文件已存在")
    
    print(f"✓ 数据目录设置完成: {data_dir.absolute()}")
    print(f"✓ 图片目录: {images_dir.absolute()}")

def main():
    """主函数"""
    print("=" * 50)
    print("图片描述匹配系统 - 服务启动器")
    print("=" * 50)
    
    # 检查依赖
    if not check_dependencies():
        sys.exit(1)
    
    # 设置数据目录
    setup_data_directory()
    
    # 启动参数
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    
    print(f"\n启动配置:")
    print(f"- 主机: {host}")
    print(f"- 端口: {port}")
    print(f"- API文档: http://localhost:{port}/docs")
    print(f"- ReDoc文档: http://localhost:{port}/redoc")
    
    print(f"\n正在启动服务...")
    print("=" * 50)
    
    try:
        uvicorn.run(
            "main:app",
            host=host,
            port=port,
            reload=True,
            log_level="info",
            access_log=True
        )
    except KeyboardInterrupt:
        print("\n服务已停止")
    except Exception as e:
        print(f"启动失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()