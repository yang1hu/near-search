#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import uvicorn
from datetime import datetime
from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from src.matcher import ImageMatcher
from src.models import (
    SearchRequest, SearchResponse, ImageResult,
    AddDescriptionRequest, ImageDescriptionResponse,
    SystemStatsResponse, UpdateMethodRequest,
    BaseResponse, HealthResponse, KeywordExtractionRequest,
    KeywordExtractionResponse, BatchDescriptionRequest,
    BatchDescriptionResponse
)

# 全局变量存储匹配器实例
matcher = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    global matcher
    print("正在启动图片描述匹配系统...")
    
    try:
        # 初始化匹配器
        matcher = ImageMatcher()
        print("系统初始化完成")
        yield
    except Exception as e:
        print(f"系统初始化失败: {e}")
        raise
    finally:
        print("系统正在关闭...")


# 创建FastAPI应用
app = FastAPI(
    title="图片描述匹配系统",
    description="基于描述词匹配图片的智能检索API",
    version="1.0.0",
    lifespan=lifespan
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_model=HealthResponse)
async def root():
    """根路径 - 健康检查"""
    return HealthResponse(
        status="running",
        version="1.0.0",
        timestamp=datetime.now().isoformat()
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """健康检查接口"""
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        timestamp=datetime.now().isoformat()
    )


@app.post("/search", response_model=SearchResponse)
async def search_images(request: SearchRequest):
    """搜索图片接口"""
    try:
        if not matcher:
            raise HTTPException(status_code=500, detail="系统未初始化")
        
        # 执行搜索
        results = matcher.search_images(
            query=request.query,
            top_k=request.top_k,
            threshold=request.threshold
        )
        
        # 转换结果格式
        image_results = []
        for result in results:
            image_results.append(ImageResult(
                image_name=result["image_name"],
                image_path=result["image_path"],
                description=result["description"],
                keywords=result["keywords"],
                similarity_score=result["similarity_score"],
                file_exists=os.path.exists(result["image_path"])
            ))
        
        return SearchResponse(
            success=True,
            message=f"找到 {len(image_results)} 张匹配的图片",
            query=request.query,
            total_results=len(image_results),
            results=image_results
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"搜索失败: {str(e)}")


@app.get("/image/{image_name}/description", response_model=ImageDescriptionResponse)
async def get_image_description(image_name: str):
    """获取图片描述接口"""
    try:
        if not matcher:
            raise HTTPException(status_code=500, detail="系统未初始化")
        
        desc_info = matcher.get_image_description(image_name)
        
        if not desc_info:
            return ImageDescriptionResponse(
                success=False,
                message="未找到该图片的描述信息",
                image_name=image_name
            )
        
        return ImageDescriptionResponse(
            success=True,
            message="获取描述成功",
            image_name=image_name,
            description_id=desc_info.get("description_id"),
            description_text=desc_info.get("description_text"),
            keywords=desc_info.get("keywords", [])
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取描述失败: {str(e)}")


@app.post("/image/description", response_model=BaseResponse)
async def add_image_description(request: AddDescriptionRequest):
    """添加图片描述接口"""
    try:
        if not matcher:
            raise HTTPException(status_code=500, detail="系统未初始化")
        
        success = matcher.add_image_description(
            image_name=request.image_name,
            description=request.description,
            keywords=request.keywords
        )
        
        if success:
            return BaseResponse(
                success=True,
                message="描述添加成功"
            )
        else:
            return BaseResponse(
                success=False,
                message="描述添加失败"
            )
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"添加描述失败: {str(e)}")


@app.get("/stats", response_model=SystemStatsResponse)
async def get_system_stats():
    """获取系统统计信息接口"""
    try:
        if not matcher:
            raise HTTPException(status_code=500, detail="系统未初始化")
        
        stats = matcher.get_statistics()
        
        return SystemStatsResponse(
            success=True,
            message="获取统计信息成功",
            stats=stats
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取统计信息失败: {str(e)}")


@app.put("/similarity-method", response_model=BaseResponse)
async def update_similarity_method(request: UpdateMethodRequest):
    """更新相似度计算方法接口"""
    try:
        if not matcher:
            raise HTTPException(status_code=500, detail="系统未初始化")
        
        matcher.update_similarity_method(request.method)
        
        return BaseResponse(
            success=True,
            message=f"相似度计算方法已更新为: {request.method}"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"更新方法失败: {str(e)}")


@app.get("/image/{image_name}")
async def get_image_file(image_name: str):
    """获取图片文件接口"""
    try:
        image_path = os.path.join("data", "images", image_name)
        
        if not os.path.exists(image_path):
            raise HTTPException(status_code=404, detail="图片文件不存在")
        
        return FileResponse(
            path=image_path,
            media_type="image/*",
            filename=image_name
        )
        
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=f"获取图片失败: {str(e)}")


@app.post("/upload", response_model=BaseResponse)
async def upload_image(file: UploadFile = File(...)):
    """上传图片接口"""
    try:
        # 检查文件类型
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="只支持图片文件")
        
        # 确保上传目录存在
        upload_dir = os.path.join("data", "images")
        os.makedirs(upload_dir, exist_ok=True)
        
        # 保存文件
        file_path = os.path.join(upload_dir, file.filename)
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        return BaseResponse(
            success=True,
            message=f"图片 {file.filename} 上传成功"
        )
        
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=f"上传失败: {str(e)}")


@app.post("/extract-keywords", response_model=KeywordExtractionResponse)
async def extract_keywords(request: KeywordExtractionRequest):
    """从文本中提取关键词接口"""
    try:
        if not matcher:
            raise HTTPException(status_code=500, detail="系统未初始化")
        
        keywords = matcher.data_processor.extract_keywords_from_text(
            request.text, 
            request.max_keywords
        )
        
        return KeywordExtractionResponse(
            success=True,
            message="关键词提取成功",
            text=request.text,
            keywords=keywords
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"关键词提取失败: {str(e)}")


@app.post("/batch-descriptions", response_model=BatchDescriptionResponse)
async def process_batch_descriptions(request: BatchDescriptionRequest):
    """批量处理描述数据接口"""
    try:
        if not matcher:
            raise HTTPException(status_code=500, detail="系统未初始化")
        
        # 将字符串列表转换为描述对象列表
        desc_objects = []
        for i, text in enumerate(request.descriptions):
            desc_obj = {
                "id": f"batch_desc_{len(matcher.descriptions) + i + 1:03d}",
                "text": text
            }
            if request.auto_generate_keywords:
                desc_obj["keywords"] = matcher.data_processor.extract_keywords_from_text(text)
            else:
                desc_obj["keywords"] = []
            desc_objects.append(desc_obj)
        
        # 添加到系统中
        matcher.descriptions.extend(desc_objects)
        matcher.data_processor.descriptions = matcher.descriptions
        matcher.data_processor.save_descriptions()
        
        return BatchDescriptionResponse(
            success=True,
            message=f"成功处理 {len(desc_objects)} 条描述",
            processed_count=len(desc_objects),
            descriptions=desc_objects
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"批量处理失败: {str(e)}")


@app.post("/generate-keywords", response_model=BaseResponse)
async def generate_keywords_for_existing():
    """为现有描述生成关键词接口"""
    try:
        if not matcher:
            raise HTTPException(status_code=500, detail="系统未初始化")
        
        updated_count = matcher.data_processor.auto_generate_keywords(force_update=True)
        
        return BaseResponse(
            success=True,
            message=f"成功为 {updated_count} 条描述生成关键词"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"生成关键词失败: {str(e)}")


@app.get("/vector-store/info")
async def get_vector_store_info():
    """获取向量库信息接口"""
    try:
        if not matcher:
            raise HTTPException(status_code=500, detail="系统未初始化")
        
        info = matcher.get_vector_store_info()
        
        return {
            "success": True,
            "message": "获取向量库信息成功",
            "data": info
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取向量库信息失败: {str(e)}")


@app.post("/vector-store/rebuild", response_model=BaseResponse)
async def rebuild_vector_index():
    """重建向量索引接口"""
    try:
        if not matcher:
            raise HTTPException(status_code=500, detail="系统未初始化")
        
        success = matcher.rebuild_vector_index()
        
        if success:
            return BaseResponse(
                success=True,
                message="向量索引重建成功"
            )
        else:
            return BaseResponse(
                success=False,
                message="向量索引重建失败"
            )
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"重建向量索引失败: {str(e)}")


@app.get("/vector-store/stats")
async def get_vector_store_stats():
    """获取向量库统计信息接口"""
    try:
        if not matcher:
            raise HTTPException(status_code=500, detail="系统未初始化")
        
        stats = matcher.similarity_calculator.get_vector_store_stats()
        
        return {
            "success": True,
            "message": "获取向量库统计成功",
            "stats": stats
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取向量库统计失败: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8001,
        reload=True,
        log_level="info"
    )