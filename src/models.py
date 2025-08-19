from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any


class SearchRequest(BaseModel):
    """搜索请求模型"""
    query: str = Field(..., description="搜索关键词", min_length=1)
    top_k: int = Field(5, description="返回结果数量", ge=1, le=20)
    threshold: float = Field(0.1, description="相似度阈值", ge=0.0, le=1.0)


class ImageResult(BaseModel):
    """图片搜索结果模型"""
    image_name: str = Field(..., description="图片文件名")
    image_path: str = Field(..., description="图片文件路径")
    description: str = Field(..., description="图片描述")
    keywords: List[str] = Field(default_factory=list, description="关键词列表")
    similarity_score: float = Field(..., description="相似度分数")
    file_exists: bool = Field(..., description="文件是否存在")


class SearchResponse(BaseModel):
    """搜索响应模型"""
    success: bool = Field(..., description="请求是否成功")
    message: str = Field(..., description="响应消息")
    query: str = Field(..., description="搜索词")
    total_results: int = Field(..., description="结果总数")
    results: List[ImageResult] = Field(default_factory=list, description="搜索结果")


class AddDescriptionRequest(BaseModel):
    """添加描述请求模型"""
    image_name: str = Field(..., description="图片文件名", min_length=1)
    description: str = Field(..., description="图片描述", min_length=1)
    keywords: Optional[List[str]] = Field(default_factory=list, description="关键词列表")


class ImageDescriptionResponse(BaseModel):
    """图片描述响应模型"""
    success: bool = Field(..., description="请求是否成功")
    message: str = Field(..., description="响应消息")
    image_name: str = Field(..., description="图片文件名")
    description_id: Optional[str] = Field(None, description="描述ID")
    description_text: Optional[str] = Field(None, description="描述文本")
    keywords: List[str] = Field(default_factory=list, description="关键词列表")


class SystemStats(BaseModel):
    """系统统计信息模型"""
    total_descriptions: int = Field(..., description="描述总数")
    total_images: int = Field(..., description="图片总数")
    similarity_method: str = Field(..., description="相似度计算方法")
    available_keywords: int = Field(..., description="可用关键词数量")


class SystemStatsResponse(BaseModel):
    """系统统计响应模型"""
    success: bool = Field(..., description="请求是否成功")
    message: str = Field(..., description="响应消息")
    stats: SystemStats = Field(..., description="统计信息")


class UpdateMethodRequest(BaseModel):
    """更新相似度方法请求模型"""
    method: str = Field(..., description="相似度计算方法", pattern="^(tfidf|sentence_transformer)$")


class BaseResponse(BaseModel):
    """基础响应模型"""
    success: bool = Field(..., description="请求是否成功")
    message: str = Field(..., description="响应消息")


class HealthResponse(BaseModel):
    """健康检查响应模型"""
    status: str = Field(..., description="服务状态")
    version: str = Field(..., description="版本信息")
    timestamp: str = Field(..., description="时间戳")


class KeywordExtractionRequest(BaseModel):
    """关键词提取请求模型"""
    text: str = Field(..., description="要提取关键词的文本", min_length=1)
    max_keywords: int = Field(5, description="最大关键词数量", ge=1, le=10)


class KeywordExtractionResponse(BaseModel):
    """关键词提取响应模型"""
    success: bool = Field(..., description="请求是否成功")
    message: str = Field(..., description="响应消息")
    text: str = Field(..., description="原始文本")
    keywords: List[str] = Field(default_factory=list, description="提取的关键词")


class BatchDescriptionRequest(BaseModel):
    """批量描述处理请求模型"""
    descriptions: List[str] = Field(..., description="描述文本列表", min_items=1)
    auto_generate_keywords: bool = Field(True, description="是否自动生成关键词")


class BatchDescriptionResponse(BaseModel):
    """批量描述处理响应模型"""
    success: bool = Field(..., description="请求是否成功")
    message: str = Field(..., description="响应消息")
    processed_count: int = Field(..., description="处理的描述数量")
    descriptions: List[Dict] = Field(default_factory=list, description="处理后的描述数据")