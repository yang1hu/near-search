#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import requests
import json
import os
from typing import Dict, List


class ImageMatcherClient:
    """图片匹配系统API客户端"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        
    def health_check(self) -> Dict:
        """健康检查"""
        response = requests.get(f"{self.base_url}/health")
        return response.json()
    
    def search_images(self, query: str, top_k: int = 5, threshold: float = 0.1) -> Dict:
        """搜索图片"""
        data = {
            "query": query,
            "top_k": top_k,
            "threshold": threshold
        }
        response = requests.post(f"{self.base_url}/search", json=data)
        return response.json()
    
    def get_image_description(self, image_name: str) -> Dict:
        """获取图片描述"""
        response = requests.get(f"{self.base_url}/image/{image_name}/description")
        return response.json()
    
    def add_image_description(self, image_name: str, description: str, keywords: List[str] = None) -> Dict:
        """添加图片描述"""
        data = {
            "image_name": image_name,
            "description": description,
            "keywords": keywords or []
        }
        response = requests.post(f"{self.base_url}/image/description", json=data)
        return response.json()
    
    def get_stats(self) -> Dict:
        """获取系统统计"""
        response = requests.get(f"{self.base_url}/stats")
        return response.json()
    
    def update_similarity_method(self, method: str) -> Dict:
        """更新相似度方法"""
        data = {"method": method}
        response = requests.put(f"{self.base_url}/similarity-method", json=data)
        return response.json()
    
    def upload_image(self, file_path: str) -> Dict:
        """上传图片"""
        if not os.path.exists(file_path):
            return {"error": "文件不存在"}
        
        with open(file_path, 'rb') as f:
            files = {'file': f}
            response = requests.post(f"{self.base_url}/upload", files=files)
        return response.json()


def demo_api_usage():
    """演示API使用"""
    print("图片描述匹配系统 API 演示")
    print("=" * 50)
    
    # 创建客户端
    client = ImageMatcherClient()
    
    try:
        # 1. 健康检查
        print("\n1. 健康检查")
        health = client.health_check()
        print(f"服务状态: {health.get('status')}")
        print(f"版本: {health.get('version')}")
        
        # 2. 获取系统统计
        print("\n2. 系统统计信息")
        stats = client.get_stats()
        if stats.get('success'):
            stats_data = stats['stats']
            print(f"描述数量: {stats_data['total_descriptions']}")
            print(f"图片数量: {stats_data['total_images']}")
            print(f"相似度方法: {stats_data['similarity_method']}")
        
        # 3. 搜索演示
        print("\n3. 搜索演示")
        search_queries = ["日落", "猫咪", "城市", "风景"]
        
        for query in search_queries:
            print(f"\n搜索: '{query}'")
            results = client.search_images(query, top_k=3, threshold=0.1)
            
            if results.get('success') and results['results']:
                print(f"找到 {results['total_results']} 张图片:")
                for i, img in enumerate(results['results'], 1):
                    print(f"  {i}. {img['image_name']} (相似度: {img['similarity_score']:.3f})")
                    print(f"     描述: {img['description']}")
                    print(f"     关键词: {', '.join(img['keywords'])}")
            else:
                print("  没有找到匹配的图片")
        
        # 4. 添加新描述演示
        print("\n4. 添加描述演示")
        new_desc_result = client.add_image_description(
            image_name="demo_image.jpg",
            description="演示用的测试图片",
            keywords=["演示", "测试", "样例"]
        )
        print(f"添加结果: {new_desc_result.get('message')}")
        
        # 5. 获取图片描述
        print("\n5. 获取图片描述")
        desc_result = client.get_image_description("demo_image.jpg")
        if desc_result.get('success'):
            print(f"图片: {desc_result['image_name']}")
            print(f"描述: {desc_result.get('description_text')}")
            print(f"关键词: {', '.join(desc_result.get('keywords', []))}")
        
        # 6. 切换相似度方法演示
        print("\n6. 相似度方法切换")
        methods = ["tfidf", "sentence_transformer"]
        for method in methods:
            result = client.update_similarity_method(method)
            print(f"切换到 {method}: {result.get('message')}")
            
            # 用新方法搜索一次
            search_result = client.search_images("美丽", top_k=2)
            if search_result.get('success'):
                print(f"  使用 {method} 找到 {search_result['total_results']} 张图片")
        
    except requests.exceptions.ConnectionError:
        print("错误: 无法连接到API服务")
        print("请确保服务已启动: python main.py")
    except Exception as e:
        print(f"发生错误: {e}")


def interactive_client():
    """交互式客户端"""
    print("图片描述匹配系统 - 交互式客户端")
    print("=" * 50)
    
    client = ImageMatcherClient()
    
    while True:
        print("\n请选择操作:")
        print("1. 搜索图片")
        print("2. 查看图片描述")
        print("3. 添加图片描述")
        print("4. 上传图片")
        print("5. 系统统计")
        print("6. 切换相似度方法")
        print("7. 运行演示")
        print("8. 退出")
        
        choice = input("\n请输入选项 (1-8): ").strip()
        
        try:
            if choice == "1":
                query = input("请输入搜索词: ").strip()
                if query:
                    top_k = int(input("返回结果数量 (默认5): ").strip() or "5")
                    threshold = float(input("相似度阈值 (默认0.1): ").strip() or "0.1")
                    
                    results = client.search_images(query, top_k, threshold)
                    print(f"\n搜索结果:")
                    if results.get('success') and results['results']:
                        for i, img in enumerate(results['results'], 1):
                            print(f"{i}. {img['image_name']} (相似度: {img['similarity_score']:.3f})")
                            print(f"   描述: {img['description']}")
                            print(f"   文件存在: {'是' if img['file_exists'] else '否'}")
                    else:
                        print("没有找到匹配的图片")
            
            elif choice == "2":
                image_name = input("请输入图片文件名: ").strip()
                if image_name:
                    result = client.get_image_description(image_name)
                    if result.get('success'):
                        print(f"\n图片: {result['image_name']}")
                        print(f"描述: {result.get('description_text')}")
                        print(f"关键词: {', '.join(result.get('keywords', []))}")
                    else:
                        print(f"错误: {result.get('message')}")
            
            elif choice == "3":
                image_name = input("请输入图片文件名: ").strip()
                description = input("请输入描述: ").strip()
                keywords_input = input("请输入关键词 (用逗号分隔): ").strip()
                
                if image_name and description:
                    keywords = [kw.strip() for kw in keywords_input.split(",") if kw.strip()]
                    result = client.add_image_description(image_name, description, keywords)
                    print(f"结果: {result.get('message')}")
            
            elif choice == "4":
                file_path = input("请输入图片文件路径: ").strip()
                if file_path:
                    result = client.upload_image(file_path)
                    print(f"上传结果: {result.get('message', result.get('error'))}")
            
            elif choice == "5":
                result = client.get_stats()
                if result.get('success'):
                    stats = result['stats']
                    print(f"\n系统统计:")
                    print(f"描述数量: {stats['total_descriptions']}")
                    print(f"图片数量: {stats['total_images']}")
                    print(f"相似度方法: {stats['similarity_method']}")
                    print(f"可用关键词: {stats['available_keywords']}")
            
            elif choice == "6":
                print("可用方法:")
                print("1. tfidf")
                print("2. sentence_transformer")
                method_choice = input("请选择 (1-2): ").strip()
                
                method = "tfidf" if method_choice == "1" else "sentence_transformer"
                result = client.update_similarity_method(method)
                print(f"结果: {result.get('message')}")
            
            elif choice == "7":
                demo_api_usage()
            
            elif choice == "8":
                print("感谢使用！")
                break
            
            else:
                print("无效选择")
                
        except requests.exceptions.ConnectionError:
            print("错误: 无法连接到API服务，请确保服务已启动")
        except ValueError as e:
            print(f"输入格式错误: {e}")
        except Exception as e:
            print(f"发生错误: {e}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        demo_api_usage()
    else:
        interactive_client()