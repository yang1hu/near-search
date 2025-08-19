import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import jieba
from typing import List, Tuple, Dict


class SimilarityCalculator:
    """相似度计算器，支持多种相似度计算方法"""
    
    def __init__(self, method: str = "tfidf"):
        self.method = method
        self.vectorizer = None
        self.sentence_model = None
        
        if method == "tfidf":
            self.vectorizer = TfidfVectorizer(
                tokenizer=self._tokenize,
                lowercase=False,
                token_pattern=None
            )
        elif method == "sentence_transformer":
            try:
                # 使用中文语义模型
                self.sentence_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
            except Exception as e:
                print(f"加载语义模型失败: {e}")
                print("回退到TF-IDF方法")
                self.method = "tfidf"
                self.vectorizer = TfidfVectorizer(
                    tokenizer=self._tokenize,
                    lowercase=False,
                    token_pattern=None
                )
    
    def _tokenize(self, text: str) -> List[str]:
        """中文分词"""
        return list(jieba.cut(text))
    
    def calculate_tfidf_similarity(self, query: str, texts: List[str]) -> List[float]:
        """使用TF-IDF计算相似度"""
        all_texts = [query] + texts
        
        # 构建TF-IDF矩阵
        tfidf_matrix = self.vectorizer.fit_transform(all_texts)
        
        # 计算查询与所有文本的余弦相似度
        query_vector = tfidf_matrix[0:1]
        text_vectors = tfidf_matrix[1:]
        
        similarities = cosine_similarity(query_vector, text_vectors)[0]
        return similarities.tolist()
    
    def calculate_semantic_similarity(self, query: str, texts: List[str]) -> List[float]:
        """使用语义模型计算相似度"""
        if self.sentence_model is None:
            return self.calculate_tfidf_similarity(query, texts)
        
        try:
            # 编码查询和文本
            query_embedding = self.sentence_model.encode([query])
            text_embeddings = self.sentence_model.encode(texts)
            
            # 计算余弦相似度
            similarities = cosine_similarity(query_embedding, text_embeddings)[0]
            return similarities.tolist()
        except Exception as e:
            print(f"语义相似度计算失败: {e}")
            return self.calculate_tfidf_similarity(query, texts)
    
    def calculate_keyword_similarity(self, query_keywords: List[str], 
                                   target_keywords: List[str]) -> float:
        """基于关键词的简单相似度计算"""
        if not query_keywords or not target_keywords:
            return 0.0
        
        query_set = set(query_keywords)
        target_set = set(target_keywords)
        
        # Jaccard相似度
        intersection = len(query_set & target_set)
        union = len(query_set | target_set)
        
        return intersection / union if union > 0 else 0.0
    
    def find_similar_descriptions(self, query: str, descriptions: List[Dict], 
                                top_k: int = 5, threshold: float = 0.1) -> List[Tuple[Dict, float]]:
        """找到与查询最相似的描述"""
        if not descriptions:
            return []
        
        # 提取描述文本
        texts = [desc["text"] for desc in descriptions]
        
        # 计算相似度
        if self.method == "sentence_transformer":
            similarities = self.calculate_semantic_similarity(query, texts)
        else:
            similarities = self.calculate_tfidf_similarity(query, texts)
        
        # 结合关键词相似度
        query_keywords = self._tokenize(query)
        combined_scores = []
        
        for i, (desc, sim_score) in enumerate(zip(descriptions, similarities)):
            keyword_sim = self.calculate_keyword_similarity(
                query_keywords, desc.get("keywords", [])
            )
            # 加权组合两种相似度
            combined_score = 0.7 * sim_score + 0.3 * keyword_sim
            combined_scores.append((desc, combined_score))
        
        # 过滤低于阈值的结果
        filtered_results = [(desc, score) for desc, score in combined_scores if score >= threshold]
        
        # 按相似度排序并返回top_k
        filtered_results.sort(key=lambda x: x[1], reverse=True)
        return filtered_results[:top_k]