import json
import os
import re
from typing import Dict, List, Tuple
from collections import Counter
import jieba
import jieba.posseg as pseg


class DataProcessor:
    """数据处理器，负责加载和处理图片描述数据"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.descriptions = []
        self.image_mappings = {}
        
    def load_descriptions(self) -> List[Dict]:
        """加载描述词数据"""
        desc_file = os.path.join(self.data_dir, "descriptions.json")
        try:
            with open(desc_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.descriptions = data.get("descriptions", [])
                print(f"成功加载 {len(self.descriptions)} 条描述数据")
                return self.descriptions
        except FileNotFoundError:
            print(f"描述文件 {desc_file} 不存在")
            return []
        except json.JSONDecodeError:
            print(f"描述文件 {desc_file} 格式错误")
            return []
    
    def scan_images(self) -> List[str]:
        """扫描图片目录，获取所有图片文件"""
        image_dir = os.path.join(self.data_dir, "images")
        if not os.path.exists(image_dir):
            os.makedirs(image_dir)
            print(f"创建图片目录: {image_dir}")
            return []
        
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'}
        images = []
        
        for filename in os.listdir(image_dir):
            if any(filename.lower().endswith(ext) for ext in image_extensions):
                images.append(filename)
        
        print(f"发现 {len(images)} 张图片")
        return images
    
    def create_mappings(self, images: List[str]) -> Dict:
        """创建图片与描述的映射关系"""
        mappings = {}
        
        # 简单的映射策略：按顺序分配描述给图片
        for i, image in enumerate(images):
            desc_index = i % len(self.descriptions) if self.descriptions else 0
            if self.descriptions:
                mappings[image] = {
                    "description_id": self.descriptions[desc_index]["id"],
                    "description_text": self.descriptions[desc_index]["text"],
                    "keywords": self.descriptions[desc_index]["keywords"]
                }
        
        self.image_mappings = mappings
        self.save_mappings()
        return mappings
    
    def save_mappings(self):
        """保存映射关系到文件"""
        mapping_file = os.path.join(self.data_dir, "mappings.json")
        with open(mapping_file, 'w', encoding='utf-8') as f:
            json.dump(self.image_mappings, f, ensure_ascii=False, indent=2)
        print(f"映射关系已保存到 {mapping_file}")
    
    def load_mappings(self) -> Dict:
        """加载已保存的映射关系"""
        mapping_file = os.path.join(self.data_dir, "mappings.json")
        try:
            with open(mapping_file, 'r', encoding='utf-8') as f:
                self.image_mappings = json.load(f)
                return self.image_mappings
        except FileNotFoundError:
            print("映射文件不存在，将创建新的映射关系")
            return {}
    
    def tokenize_text(self, text: str) -> List[str]:
        """中文文本分词"""
        return list(jieba.cut(text))
    
    def extract_keywords_from_text(self, text: str, max_keywords: int = 5) -> List[str]:
        """从文本中自动提取关键词"""
        if not text.strip():
            return []
        
        # 停用词列表
        stop_words = {
            '的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个', '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好', '自己', '这', '那', '里', '就是', '还', '把', '比', '或者', '虽然', '因为', '所以', '但是', '如果', '这样', '那样', '怎么', '什么', '哪里', '为什么', '怎样', '多少', '第一', '可以', '应该', '能够', '已经', '正在', '将要'
        }
        
        # 使用词性标注进行分词
        words = pseg.cut(text)
        
        # 筛选有意义的词汇（名词、动词、形容词等）
        meaningful_words = []
        for word, flag in words:
            # 过滤条件：
            # 1. 长度大于1
            # 2. 不是停用词
            # 3. 是有意义的词性（名词n、动词v、形容词a等）
            # 4. 不是纯数字或标点
            if (len(word) > 1 and 
                word not in stop_words and 
                flag.startswith(('n', 'v', 'a', 'i', 'l')) and  # 名词、动词、形容词、成语、习语
                not re.match(r'^[\d\W]+$', word)):
                meaningful_words.append(word)
        
        # 统计词频并返回最高频的关键词
        if meaningful_words:
            word_freq = Counter(meaningful_words)
            keywords = [word for word, freq in word_freq.most_common(max_keywords)]
            return keywords
        
        # 如果没有找到有意义的词，则使用简单分词
        simple_words = [word for word in jieba.cut(text) 
                       if len(word) > 1 and word not in stop_words]
        return simple_words[:max_keywords]
    
    def auto_generate_keywords(self, force_update: bool = False) -> int:
        """为没有关键词的描述自动生成关键词"""
        updated_count = 0
        
        for desc in self.descriptions:
            # 如果没有关键词或者强制更新
            if not desc.get("keywords") or force_update:
                keywords = self.extract_keywords_from_text(desc["text"])
                desc["keywords"] = keywords
                updated_count += 1
                print(f"为描述 '{desc['text'][:20]}...' 生成关键词: {keywords}")
        
        if updated_count > 0:
            # 保存更新后的描述数据
            self.save_descriptions()
            print(f"共为 {updated_count} 条描述生成了关键词")
        
        return updated_count
    
    def save_descriptions(self):
        """保存描述数据到文件"""
        desc_file = os.path.join(self.data_dir, "descriptions.json")
        data = {"descriptions": self.descriptions}
        with open(desc_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"描述数据已保存到 {desc_file}")
    
    def process_simple_descriptions(self, descriptions_data: List[Dict]) -> List[Dict]:
        """处理简单的描述数据，自动生成关键词"""
        processed_descriptions = []
        
        for i, desc_data in enumerate(descriptions_data):
            # 如果只有text字段，自动生成其他字段
            if isinstance(desc_data, dict) and "text" in desc_data:
                processed_desc = {
                    "id": desc_data.get("id", f"desc_{i+1:03d}"),
                    "text": desc_data["text"],
                    "keywords": desc_data.get("keywords", self.extract_keywords_from_text(desc_data["text"]))
                }
            elif isinstance(desc_data, str):
                # 如果直接是字符串，转换为完整格式
                processed_desc = {
                    "id": f"desc_{i+1:03d}",
                    "text": desc_data,
                    "keywords": self.extract_keywords_from_text(desc_data)
                }
            else:
                processed_desc = desc_data
            
            processed_descriptions.append(processed_desc)
            
        return processed_descriptions
    
    def get_all_keywords(self) -> List[str]:
        """获取所有关键词"""
        all_keywords = set()
        for desc in self.descriptions:
            all_keywords.update(desc.get("keywords", []))
            # 添加描述文本的分词结果
            tokens = self.tokenize_text(desc["text"])
            all_keywords.update(tokens)
        return list(all_keywords)