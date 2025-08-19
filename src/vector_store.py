import os
import json
import pickle
import numpy as np
import faiss
import h5py
from typing import List, Dict, Tuple, Optional, Any
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
import jieba
from datetime import datetime


class VectorStore:
    """向量存储和检索系统"""
    
    def __init__(self, store_dir: str = "data/vectors", embedding_dim: int = 384):
        self.store_dir = store_dir
        self.embedding_dim = embedding_dim
        
        # 确保存储目录存在
        os.makedirs(store_dir, exist_ok=True)
        
        # FAISS索引
        self.faiss_index = None
        self.index_file = os.path.join(store_dir, "faiss_index.bin")
        
        # 向量和元数据存储
        self.vectors_file = os.path.join(store_dir, "vectors.h5")
        self.metadata_file = os.path.join(store_dir, "metadata.json")
        
        # 内存中的数据
        self.vectors = []
        self.metadata = []
        self.id_to_index = {}
        
        # 加载已有数据
        self.load_store()
    
    def load_store(self):
        """加载已保存的向量库"""
        try:
            # 加载FAISS索引
            if os.path.exists(self.index_file):
                self.faiss_index = faiss.read_index(self.index_file)
                print(f"✓ 加载FAISS索引: {self.faiss_index.ntotal} 个向量")
            else:
                # 创建新的FAISS索引 (使用内积搜索，适合归一化向量)
                self.faiss_index = faiss.IndexFlatIP(self.embedding_dim)
                print("✓ 创建新的FAISS索引")
            
            # 加载元数据
            if os.path.exists(self.metadata_file):
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    self.metadata = json.load(f)
                    
                # 重建ID到索引的映射
                self.id_to_index = {
                    item['id']: idx for idx, item in enumerate(self.metadata)
                }
                print(f"✓ 加载元数据: {len(self.metadata)} 条记录")
            
            # 加载向量数据到内存（用于快速访问）
            if os.path.exists(self.vectors_file):
                with h5py.File(self.vectors_file, 'r') as f:
                    if 'vectors' in f:
                        self.vectors = f['vectors'][:]
                        print(f"✓ 加载向量数据: {self.vectors.shape}")
                        
        except Exception as e:
            print(f"加载向量库时出错: {e}")
            self._initialize_empty_store()
    
    def _initialize_empty_store(self):
        """初始化空的向量库"""
        self.faiss_index = faiss.IndexFlatIP(self.embedding_dim)
        self.vectors = []
        self.metadata = []
        self.id_to_index = {}
        print("✓ 初始化空向量库")
    
    def add_vectors(self, vectors: np.ndarray, metadata_list: List[Dict]) -> List[int]:
        """添加向量到存储库"""
        if len(vectors) != len(metadata_list):
            raise ValueError("向量数量与元数据数量不匹配")
        
        # 归一化向量（用于余弦相似度）
        vectors_normalized = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
        
        # 添加到FAISS索引
        self.faiss_index.add(vectors_normalized.astype(np.float32))
        
        # 更新内存数据
        if len(self.vectors) == 0:
            self.vectors = vectors_normalized
        else:
            self.vectors = np.vstack([self.vectors, vectors_normalized])
        
        # 添加元数据
        start_index = len(self.metadata)
        indices = []
        
        for i, meta in enumerate(metadata_list):
            index = start_index + i
            meta['index'] = index
            meta['created_at'] = datetime.now().isoformat()
            
            self.metadata.append(meta)
            self.id_to_index[meta['id']] = index
            indices.append(index)
        
        print(f"✓ 添加 {len(vectors)} 个向量到存储库")
        return indices
    
    def search_similar(self, query_vector: np.ndarray, top_k: int = 10, 
                      threshold: float = 0.0) -> List[Tuple[Dict, float]]:
        """搜索相似向量"""
        if self.faiss_index.ntotal == 0:
            return []
        
        # 归一化查询向量
        query_normalized = query_vector / np.linalg.norm(query_vector)
        query_normalized = query_normalized.reshape(1, -1).astype(np.float32)
        
        # 使用FAISS搜索
        scores, indices = self.faiss_index.search(query_normalized, min(top_k * 2, self.faiss_index.ntotal))
        
        # 过滤结果并组装返回数据
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and score >= threshold:  # FAISS可能返回-1表示无效索引
                if idx < len(self.metadata):
                    results.append((self.metadata[idx], float(score)))
        
        # 按分数排序并返回top_k
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
    
    def get_vector_by_id(self, vector_id: str) -> Optional[Tuple[np.ndarray, Dict]]:
        """根据ID获取向量和元数据"""
        if vector_id in self.id_to_index:
            index = self.id_to_index[vector_id]
            if index < len(self.vectors):
                return self.vectors[index], self.metadata[index]
        return None
    
    def update_metadata(self, vector_id: str, new_metadata: Dict) -> bool:
        """更新向量的元数据"""
        if vector_id in self.id_to_index:
            index = self.id_to_index[vector_id]
            if index < len(self.metadata):
                # 保留一些系统字段
                new_metadata['id'] = vector_id
                new_metadata['index'] = index
                new_metadata['updated_at'] = datetime.now().isoformat()
                
                self.metadata[index] = new_metadata
                return True
        return False
    
    def delete_vector(self, vector_id: str) -> bool:
        """删除向量（标记为删除，不实际删除以保持索引一致性）"""
        if vector_id in self.id_to_index:
            index = self.id_to_index[vector_id]
            if index < len(self.metadata):
                self.metadata[index]['deleted'] = True
                self.metadata[index]['deleted_at'] = datetime.now().isoformat()
                return True
        return False
    
    def save_store(self):
        """保存向量库到磁盘"""
        try:
            # 保存FAISS索引
            faiss.write_index(self.faiss_index, self.index_file)
            
            # 保存元数据
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(self.metadata, f, ensure_ascii=False, indent=2)
            
            # 保存向量数据
            if len(self.vectors) > 0:
                with h5py.File(self.vectors_file, 'w') as f:
                    f.create_dataset('vectors', data=self.vectors)
            
            print(f"✓ 向量库已保存: {len(self.metadata)} 条记录")
            return True
            
        except Exception as e:
            print(f"保存向量库失败: {e}")
            return False
    
    def get_stats(self) -> Dict:
        """获取向量库统计信息"""
        active_count = sum(1 for meta in self.metadata if not meta.get('deleted', False))
        
        return {
            'total_vectors': len(self.metadata),
            'active_vectors': active_count,
            'deleted_vectors': len(self.metadata) - active_count,
            'embedding_dimension': self.embedding_dim,
            'index_type': type(self.faiss_index).__name__,
            'store_size_mb': self._get_store_size()
        }
    
    def _get_store_size(self) -> float:
        """计算存储大小（MB）"""
        total_size = 0
        for file_path in [self.index_file, self.metadata_file, self.vectors_file]:
            if os.path.exists(file_path):
                total_size += os.path.getsize(file_path)
        return round(total_size / (1024 * 1024), 2)
    
    def rebuild_index(self):
        """重建索引（清理已删除的向量）"""
        print("开始重建向量索引...")
        
        # 收集未删除的向量和元数据
        active_vectors = []
        active_metadata = []
        new_id_to_index = {}
        
        for i, meta in enumerate(self.metadata):
            if not meta.get('deleted', False):
                active_vectors.append(self.vectors[i])
                
                # 更新索引
                new_index = len(active_metadata)
                meta['index'] = new_index
                active_metadata.append(meta)
                new_id_to_index[meta['id']] = new_index
        
        if active_vectors:
            # 重建FAISS索引
            self.faiss_index = faiss.IndexFlatIP(self.embedding_dim)
            active_vectors_array = np.array(active_vectors)
            self.faiss_index.add(active_vectors_array.astype(np.float32))
            
            # 更新内存数据
            self.vectors = active_vectors_array
            self.metadata = active_metadata
            self.id_to_index = new_id_to_index
            
            print(f"✓ 索引重建完成: {len(active_metadata)} 个活跃向量")
        else:
            self._initialize_empty_store()
            print("✓ 重建为空索引")


class EnhancedSimilarityCalculator:
    """增强的相似度计算器，集成向量库"""
    
    def __init__(self, method: str = "tfidf", store_dir: str = "data/vectors"):
        self.method = method
        self.vector_store = None
        self.sentence_model = None
        self.vectorizer = None
        
        # 初始化模型
        if method == "sentence_transformer":
            try:
                self.sentence_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
                # 获取模型的嵌入维度
                embedding_dim = self.sentence_model.get_sentence_embedding_dimension()
                self.vector_store = VectorStore(store_dir, embedding_dim)
                print(f"✓ 语义模型加载成功，嵌入维度: {embedding_dim}")
            except Exception as e:
                print(f"加载语义模型失败: {e}")
                print("回退到TF-IDF方法")
                self.method = "tfidf"
        
        if method == "tfidf" or self.sentence_model is None:
            self.vectorizer = TfidfVectorizer(
                tokenizer=self._tokenize,
                lowercase=False,
                token_pattern=None
            )
    
    def _tokenize(self, text: str) -> List[str]:
        """中文分词"""
        return list(jieba.cut(text))
    
    def build_vector_index(self, descriptions: List[Dict]) -> bool:
        """构建向量索引"""
        if self.method != "sentence_transformer" or not self.sentence_model:
            print("当前方法不支持向量索引")
            return False
        
        if not descriptions:
            print("没有描述数据用于构建索引")
            return False
        
        print(f"开始构建向量索引，共 {len(descriptions)} 条描述...")
        
        # 提取文本并生成向量
        texts = [desc["text"] for desc in descriptions]
        vectors = self.sentence_model.encode(texts, show_progress_bar=True)
        
        # 准备元数据
        metadata_list = []
        for desc in descriptions:
            metadata_list.append({
                'id': desc['id'],
                'text': desc['text'],
                'keywords': desc.get('keywords', []),
                'type': 'description'
            })
        
        # 添加到向量库
        indices = self.vector_store.add_vectors(vectors, metadata_list)
        
        # 保存到磁盘
        self.vector_store.save_store()
        
        print(f"✓ 向量索引构建完成，添加了 {len(indices)} 个向量")
        return True
    
    def search_similar_vectors(self, query: str, top_k: int = 10, 
                             threshold: float = 0.1) -> List[Tuple[Dict, float]]:
        """使用向量库搜索相似内容"""
        if self.method != "sentence_transformer" or not self.sentence_model:
            return []
        
        # 生成查询向量
        query_vector = self.sentence_model.encode([query])[0]
        
        # 在向量库中搜索
        results = self.vector_store.search_similar(
            query_vector, top_k=top_k, threshold=threshold
        )
        
        # 过滤已删除的项目
        filtered_results = [
            (meta, score) for meta, score in results 
            if not meta.get('deleted', False)
        ]
        
        return filtered_results
    
    def add_description_to_index(self, description: Dict) -> bool:
        """添加新描述到向量索引"""
        if self.method != "sentence_transformer" or not self.sentence_model:
            return False
        
        # 生成向量
        vector = self.sentence_model.encode([description["text"]])[0:1]
        
        # 准备元数据
        metadata = {
            'id': description['id'],
            'text': description['text'],
            'keywords': description.get('keywords', []),
            'type': 'description'
        }
        
        # 添加到向量库
        self.vector_store.add_vectors(vector, [metadata])
        self.vector_store.save_store()
        
        return True
    
    def get_vector_store_stats(self) -> Dict:
        """获取向量库统计信息"""
        if self.vector_store:
            return self.vector_store.get_stats()
        return {}
    
    def rebuild_vector_index(self) -> bool:
        """重建向量索引"""
        if self.vector_store:
            self.vector_store.rebuild_index()
            self.vector_store.save_store()
            return True
        return False