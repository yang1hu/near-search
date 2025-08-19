import os
from typing import List, Dict, Tuple
from .data_processor import DataProcessor
from .similarity import SimilarityCalculator


class ImageMatcher:
    """图片匹配引擎，根据描述词匹配相应图片"""
    
    def __init__(self, data_dir: str = "data", similarity_method: str = "tfidf"):
        self.data_processor = DataProcessor(data_dir)
        self.similarity_calculator = SimilarityCalculator(similarity_method)
        self.descriptions = []
        self.image_mappings = {}
        
        # 初始化数据
        self.initialize()
    
    def initialize(self):
        """初始化系统数据"""
        print("正在初始化图片匹配系统...")
        
        # 加载描述数据
        self.descriptions = self.data_processor.load_descriptions()
        
        # 自动为没有关键词的描述生成关键词
        if self.descriptions:
            self.data_processor.auto_generate_keywords()
            self.descriptions = self.data_processor.descriptions  # 更新引用
        
        # 扫描图片
        images = self.data_processor.scan_images()
        
        # 加载或创建映射关系
        self.image_mappings = self.data_processor.load_mappings()
        if not self.image_mappings and images:
            self.image_mappings = self.data_processor.create_mappings(images)
        
        print(f"系统初始化完成，共有 {len(self.image_mappings)} 个图片-描述映射")
    
    def search_images(self, query: str, top_k: int = 5, 
                     threshold: float = 0.1) -> List[Dict]:
        """根据查询词搜索匹配的图片"""
        if not self.descriptions:
            print("没有可用的描述数据")
            return []
        
        print(f"正在搜索与 '{query}' 相关的图片...")
        
        # 找到相似的描述
        similar_descriptions = self.similarity_calculator.find_similar_descriptions(
            query, self.descriptions, top_k, threshold
        )
        
        if not similar_descriptions:
            print("没有找到匹配的描述")
            return []
        
        # 根据描述找到对应的图片
        results = []
        for desc, similarity_score in similar_descriptions:
            desc_id = desc["id"]
            
            # 找到使用这个描述的图片
            matching_images = [
                img_name for img_name, mapping in self.image_mappings.items()
                if mapping["description_id"] == desc_id
            ]
            
            for img_name in matching_images:
                results.append({
                    "image_name": img_name,
                    "image_path": os.path.join(self.data_processor.data_dir, "images", img_name),
                    "description": desc["text"],
                    "keywords": desc.get("keywords", []),
                    "similarity_score": similarity_score
                })
        
        # 按相似度排序
        results.sort(key=lambda x: x["similarity_score"], reverse=True)
        
        print(f"找到 {len(results)} 张匹配的图片")
        return results[:top_k]
    
    def get_image_description(self, image_name: str) -> Dict:
        """获取指定图片的描述信息"""
        if image_name in self.image_mappings:
            return self.image_mappings[image_name]
        return {}
    
    def add_image_description(self, image_name: str, description: str, 
                            keywords: List[str] = None) -> bool:
        """为图片添加新的描述"""
        try:
            # 生成新的描述ID
            desc_id = f"desc_{len(self.descriptions) + 1:03d}"
            
            # 添加到描述列表
            new_desc = {
                "id": desc_id,
                "text": description,
                "keywords": keywords or []
            }
            self.descriptions.append(new_desc)
            
            # 更新映射关系
            self.image_mappings[image_name] = {
                "description_id": desc_id,
                "description_text": description,
                "keywords": keywords or []
            }
            
            # 保存更新
            self.data_processor.descriptions = self.descriptions
            self.data_processor.image_mappings = self.image_mappings
            self.data_processor.save_mappings()
            
            print(f"成功为图片 {image_name} 添加描述")
            return True
            
        except Exception as e:
            print(f"添加描述失败: {e}")
            return False
    
    def update_similarity_method(self, method: str):
        """更新相似度计算方法"""
        self.similarity_calculator = SimilarityCalculator(method)
        print(f"相似度计算方法已更新为: {method}")
    
    def get_statistics(self) -> Dict:
        """获取系统统计信息"""
        return {
            "total_descriptions": len(self.descriptions),
            "total_images": len(self.image_mappings),
            "similarity_method": self.similarity_calculator.method,
            "available_keywords": len(self.data_processor.get_all_keywords())
        }