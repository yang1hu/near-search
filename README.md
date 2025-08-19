# 图片描述匹配系统

基于描述词匹配图片的智能检索API系统。

## 功能特性
- 图片与描述词关系建立
- 基于语义相似度的描述词匹配
- 支持中文文本处理
- 灵活的相似度阈值配置
- RESTful API接口
- 图片上传和管理
- 自动API文档生成

## 项目结构
```
├── data/                   # 数据目录
│   ├── images/            # 图片文件
│   ├── descriptions.json  # 描述词数据
│   └── mappings.json      # 图片-描述关系映射
├── src/                   # 源代码
│   ├── data_processor.py  # 数据处理模块
│   ├── similarity.py      # 相似度计算模块
│   ├── matcher.py         # 匹配引擎
│   └── models.py          # API数据模型
├── main.py                # FastAPI应用入口
└── requirements.txt       # 依赖包
```

## 快速开始

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 准备数据
将图片文件放入 `data/images/` 目录，系统会自动扫描并建立映射关系。

### 3. 启动服务
```bash
python main.py
```

服务将在 `http://localhost:8000` 启动。

### 4. 访问API文档
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## API接口说明

### 核心接口

#### 1. 搜索图片
```http
POST /search
Content-Type: application/json

{
  "query": "日落风景",
  "top_k": 5,
  "threshold": 0.1
}
```

#### 2. 获取图片描述
```http
GET /image/{image_name}/description
```

#### 3. 添加图片描述
```http
POST /image/description
Content-Type: application/json

{
  "image_name": "sunset.jpg",
  "description": "美丽的日落风景",
  "keywords": ["日落", "风景", "橙色"]
}
```

#### 4. 上传图片
```http
POST /upload
Content-Type: multipart/form-data

file: [图片文件]
```

#### 5. 获取图片文件
```http
GET /image/{image_name}
```

#### 6. 系统统计
```http
GET /stats
```

#### 7. 更新相似度方法
```http
PUT /similarity-method
Content-Type: application/json

{
  "method": "tfidf"  // 或 "sentence_transformer"
}
```

### 使用示例

#### Python客户端示例
```python
import requests

# 搜索图片
response = requests.post("http://localhost:8000/search", json={
    "query": "可爱的猫咪",
    "top_k": 3,
    "threshold": 0.2
})

results = response.json()
print(f"找到 {results['total_results']} 张图片")

for img in results['results']:
    print(f"图片: {img['image_name']}")
    print(f"描述: {img['description']}")
    print(f"相似度: {img['similarity_score']:.3f}")
```

#### cURL示例
```bash
# 搜索图片
curl -X POST "http://localhost:8000/search" \
  -H "Content-Type: application/json" \
  -d '{"query": "城市夜景", "top_k": 5}'

# 上传图片
curl -X POST "http://localhost:8000/upload" \
  -F "file=@/path/to/image.jpg"
```

## 相似度算法

系统支持两种相似度计算方法：

1. **TF-IDF** (默认)
   - 基于词频-逆文档频率的传统方法
   - 计算速度快，资源占用少
   - 适合关键词匹配

2. **Sentence Transformer**
   - 基于预训练语义模型
   - 理解语义相似性更准确
   - 支持多语言，特别优化中文

## 配置说明

### 描述数据格式 (data/descriptions.json)
```json
{
  "descriptions": [
    {
      "id": "desc_001",
      "text": "美丽的日落风景，橙色天空",
      "keywords": ["日落", "风景", "橙色", "天空", "美丽"]
    }
  ]
}
```

### 环境变量
- `HOST`: 服务监听地址 (默认: 0.0.0.0)
- `PORT`: 服务端口 (默认: 8000)
- `DATA_DIR`: 数据目录路径 (默认: data)

## 部署建议

### Docker部署
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 生产环境
```bash
# 使用Gunicorn部署
pip install gunicorn
gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```