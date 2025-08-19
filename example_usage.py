#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import requests
import json
import time
from pathlib import Path

# API基础URL
BASE_URL = "http://localhost:8000"

def test_health():
    """测试健康检查"""
    print("1. 测试健康检查...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code == 200:
            data = response.json()
            print(f"✓ 服务状态: {data['status']}")
            print(f"✓ 版本: {data['version']}")
            return True
        else:
            print(f"✗ 健康检查失败: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("✗ 无法连接到服务，请确保服务已启动")
        return False

def test_stats():
    """测试系统统计"""
    print("\n2. 获取系统统计...")
    try:
        response = requests.get(f"{BASE_URL}/stats")
        if response.status_code == 200:
            data = response.json()
            stats = data['stats']
            print(f"✓ 描述数量: {stats['total_descriptions']}")
            print(f"✓ 图片数量: {stats['total_images']}")
            print(f"✓ 相似度方法: {stats['similarity_method']}")
            return True
        else:
            print(f"✗ 获取统计失败: {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ 请求失败: {e}")
        return False

def test_search():
    """测试图片搜索"""
    print("\n3. 测试图片搜索...")
    
    test_queries = [
        {"query": "日落", "top_k": 3, "threshold": 0.1},
        {"query": "猫咪", "top_k": 2, "threshold": 0.2},
        {"query": "城市夜景", "top_k": 5, "threshold": 0.1}
    ]
    
    for i, search_data in enumerate(test_queries, 1):
        print(f"\n  测试 {i}: 搜索 '{search_data['query']}'")
        try:
            response = requests.post(f"{BASE_URL}/search", json=search_data)
            if response.status_code == 200:
                data = response.json()
                print(f"  ✓ 找到 {data['total_results']} 张图片")
                
                for j, result in enumerate(data['results'][:2], 1):  # 只显示前2个结果
                    print(f"    {j}. {result['image_name']}")
                    print(f"       描述: {result['description']}")
                    print(f"       相似度: {result['similarity_score']:.3f}")
                    print(f"       文件存在: {'是' if result['file_exists'] else '否'}")
            else:
                print(f"  ✗ 搜索失败: {response.status_code}")
                print(f"  错误: {response.text}")
        except Exception as e:
            print(f"  ✗ 请求失败: {e}")

def test_add_description():
    """测试添加描述"""
    print("\n4. 测试添加图片描述...")
    
    test_data = {
        "image_name": "test_image.jpg",
        "description": "这是一个测试图片描述",
        "keywords": ["测试", "图片", "描述"]
    }
    
    try:
        response = requests.post(f"{BASE_URL}/image/description", json=test_data)
        if response.status_code == 200:
            data = response.json()
            if data['success']:
                print(f"✓ {data['message']}")
            else:
                print(f"✗ {data['message']}")
        else:
            print(f"✗ 添加描述失败: {response.status_code}")
            print(f"错误: {response.text}")
    except Exception as e:
        print(f"✗ 请求失败: {e}")

def test_get_description():
    """测试获取图片描述"""
    print("\n5. 测试获取图片描述...")
    
    image_name = "test_image.jpg"
    try:
        response = requests.get(f"{BASE_URL}/image/{image_name}/description")
        if response.status_code == 200:
            data = response.json()
            if data['success']:
                print(f"✓ 图片: {data['image_name']}")
                print(f"✓ 描述: {data['description_text']}")
                print(f"✓ 关键词: {', '.join(data['keywords'])}")
            else:
                print(f"✗ {data['message']}")
        else:
            print(f"✗ 获取描述失败: {response.status_code}")
    except Exception as e:
        print(f"✗ 请求失败: {e}")

def test_similarity_method():
    """测试切换相似度方法"""
    print("\n6. 测试切换相似度方法...")
    
    methods = ["tfidf", "sentence_transformer"]
    
    for method in methods:
        print(f"  切换到: {method}")
        try:
            response = requests.put(f"{BASE_URL}/similarity-method", json={"method": method})
            if response.status_code == 200:
                data = response.json()
                print(f"  ✓ {data['message']}")
            else:
                print(f"  ✗ 切换失败: {response.status_code}")
        except Exception as e:
            print(f"  ✗ 请求失败: {e}")
        
        time.sleep(1)  # 等待一秒

def upload_test_image():
    """上传测试图片"""
    print("\n7. 测试图片上传...")
    
    # 创建一个简单的测试文件
    test_file_path = Path("test_image.txt")
    test_file_path.write_text("这是一个测试文件，模拟图片上传")
    
    try:
        with open(test_file_path, 'rb') as f:
            files = {'file': ('test_image.jpg', f, 'image/jpeg')}
            response = requests.post(f"{BASE_URL}/upload", files=files)
            
        if response.status_code == 200:
            data = response.json()
            if data['success']:
                print(f"✓ {data['message']}")
            else:
                print(f"✗ {data['message']}")
        else:
            print(f"✗ 上传失败: {response.status_code}")
            print(f"错误: {response.text}")
    except Exception as e:
        print(f"✗ 上传失败: {e}")
    finally:
        # 清理测试文件
        if test_file_path.exists():
            test_file_path.unlink()

def test_vector_store():
    """测试向量库功能"""
    print("\n8. 测试向量库功能...")
    
    # 获取向量库信息
    print("  8.1 获取向量库信息")
    try:
        response = requests.get(f"{BASE_URL}/vector-store/info")
        if response.status_code == 200:
            data = response.json()
            if data['success']:
                info = data['data']
                print(f"  ✓ 向量库启用: {info.get('enabled', False)}")
                if info.get('enabled'):
                    stats = info.get('stats', {})
                    print(f"  ✓ 总向量数: {stats.get('total_vectors', 0)}")
                    print(f"  ✓ 活跃向量数: {stats.get('active_vectors', 0)}")
                    print(f"  ✓ 嵌入维度: {stats.get('embedding_dimension', 0)}")
                    print(f"  ✓ 存储大小: {stats.get('store_size_mb', 0)} MB")
            else:
                print(f"  ✗ {data['message']}")
        else:
            print(f"  ✗ 获取向量库信息失败: {response.status_code}")
    except Exception as e:
        print(f"  ✗ 请求失败: {e}")
    
    # 获取向量库统计
    print("\n  8.2 获取向量库统计")
    try:
        response = requests.get(f"{BASE_URL}/vector-store/stats")
        if response.status_code == 200:
            data = response.json()
            if data['success']:
                stats = data['stats']
                print(f"  ✓ 向量库统计获取成功")
                for key, value in stats.items():
                    print(f"    {key}: {value}")
            else:
                print(f"  ✗ {data['message']}")
        else:
            print(f"  ✗ 获取统计失败: {response.status_code}")
    except Exception as e:
        print(f"  ✗ 请求失败: {e}")
    
    # 重建向量索引
    print("\n  8.3 重建向量索引")
    try:
        response = requests.post(f"{BASE_URL}/vector-store/rebuild")
        if response.status_code == 200:
            data = response.json()
            if data['success']:
                print(f"  ✓ {data['message']}")
            else:
                print(f"  ✗ {data['message']}")
        else:
            print(f"  ✗ 重建索引失败: {response.status_code}")
    except Exception as e:
        print(f"  ✗ 请求失败: {e}")

def test_performance_comparison():
    """测试性能对比"""
    print("\n9. 性能对比测试...")
    
    test_queries = ["美丽风景", "可爱动物", "现代建筑", "自然景观", "城市夜景"]
    
    for method in ["tfidf", "sentence_transformer"]:
        print(f"\n  测试方法: {method}")
        
        # 切换方法
        try:
            response = requests.put(f"{BASE_URL}/similarity-method", json={"method": method})
            if response.status_code == 200:
                print(f"  ✓ 已切换到 {method}")
            else:
                print(f"  ✗ 切换方法失败")
                continue
        except Exception as e:
            print(f"  ✗ 切换方法失败: {e}")
            continue
        
        # 测试搜索性能
        total_time = 0
        for query in test_queries:
            start_time = time.time()
            try:
                response = requests.post(f"{BASE_URL}/search", json={
                    "query": query,
                    "top_k": 3,
                    "threshold": 0.1
                })
                end_time = time.time()
                
                if response.status_code == 200:
                    data = response.json()
                    search_time = end_time - start_time
                    total_time += search_time
                    print(f"    '{query}': {search_time:.3f}s, 找到 {data['total_results']} 个结果")
                else:
                    print(f"    '{query}': 搜索失败")
            except Exception as e:
                print(f"    '{query}': 请求失败 - {e}")
        
        avg_time = total_time / len(test_queries) if test_queries else 0
        print(f"  平均搜索时间: {avg_time:.3f}s")

def main():
    """主测试函数"""
    print("=" * 60)
    print("图片描述匹配系统 - API测试")
    print("=" * 60)
    print("请确保服务已启动 (python start_server.py)")
    print("=" * 60)
    
    # 等待用户确认
    input("按回车键开始测试...")
    
    # 执行测试
    if not test_health():
        print("\n服务未启动或无法访问，测试终止")
        return
    
    test_stats()
    test_search()
    test_add_description()
    test_get_description()
    test_similarity_method()
    upload_test_image()
    test_vector_store()
    test_performance_comparison()
    
    print("\n" + "=" * 60)
    print("测试完成！")
    print("你可以访问以下地址查看API文档:")
    print(f"- Swagger UI: {BASE_URL}/docs")
    print(f"- ReDoc: {BASE_URL}/redoc")
    print("\n向量库相关接口:")
    print(f"- 向量库信息: {BASE_URL}/vector-store/info")
    print(f"- 向量库统计: {BASE_URL}/vector-store/stats")
    print(f"- 重建索引: {BASE_URL}/vector-store/rebuild")
    print("=" * 60)

if __name__ == "__main__":
    main()