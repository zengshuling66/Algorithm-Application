#cockpit-rag:day6 是构建产物/镜像，装有 Linux 用户空间、Python 3.11、CPU PyTorch、依赖和项目代码。
#cockpit-rag-day6-api 是用这个镜像启动的一个容器。
#docker build ... . 的 . 指定 Docker 能从哪里取文件。
# .dockerignore 先过滤，Dockerfile 再复制和安装。
#`RUN` 在构建时执行，`CMD` 在容器启动时执行。
#Qdrant 和 Hugging Face 缓存用命名卷，删除容器不会自动删除卷；Docker 镜像和命名卷由 Ubuntu 里的 Docker Engine 管理。
#原始 PDF 已经在宿主机上，用只读绑定挂载给入库容器；入库/模型权重下载用的 docker run --rm 是一次性容器，退出后会被自动移除。
#换密钥只重建容器，命名卷重新挂载复用；改镜像内的代码，需要重新构建镜像再启动容器；换手册要核对资料身份和版本，重新解析、入库并刷新检索器。
#镜像管程序，卷管数据，环境变量管本次运行配置。
#容器里的 /app 是构建镜像时复制进去的代码。在 Windows 或 Ubuntu 修改项目源码，两边都能立即看到，但已经构建好的镜像和正在运行的容器不会自动更新。
#docker run 创建容器时就确定了挂载关系：从哪里挂、挂到容器哪个路径、是否只读。`docker start` 或 docker restart 只会沿用这套配置，不会添加新挂载。
#所以要给现有容器增减/修改一个挂载，也要重新创建容器，但不用重新构建镜像；但在已经绑定挂载的目录里新增文件则不需要，因为容器已经能访问这个目录。

# 与本地开发环境保持 Python 3.11 一致，不依赖宿主机上的 Conda。
FROM python:3.11-slim-bookworm

# HF_HOME 是运行时的模型缓存挂载点，构建镜像时不下载权重。
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/home/rag/.cache/huggingface

WORKDIR /app

# 先装官方 CPU 版 PyTorch，避免引入不需要的 CUDA 依赖。
# 先安装依赖，再复制源码。只改业务代码时，可以复用依赖安装层。
COPY requirements.txt ./
RUN python -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.8.0 \
    && python -m pip install --no-cache-dir -r requirements.txt \
    && python -m pip check

# 只复制经 .dockerignore 放行的文件，不把本地库和评测产物打进镜像。
COPY . ./

# 创建非 root 用户，并为本地 Qdrant 和模型缓存准备可写的挂载点。
RUN useradd --create-home --uid 10001 --user-group rag \
    && install -d -o rag -g rag /app/.qdrant /home/rag/.cache/huggingface

USER rag
# 只声明容器内的端口。宿主机用哪个端口，由 docker run -p 指定。
EXPOSE 8001

# 监听容器内所有网卡，才能接收端口转发的请求。
# 保留一个 worker，避免多个服务进程同时打开同一个本地 Qdrant 目录。
# DeepSeek 密钥在运行时注入，不写进镜像。
CMD ["python", "-m", "uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8001", "--workers", "1"]