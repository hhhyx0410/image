# 基础镜像
FROM ubuntu:22.04

# 更新系统包并安装Python和pip
RUN apt-get update && apt-get install -y python3 python3-pip

# 设置工作目录
WORKDIR /app

# 将当前目录的内容复制到容器的/app目录中
COPY . /app

# 安装Flask应用所需的依赖项
RUN pip3 install -r requirements.txt

# 暴露Flask应用的端口
EXPOSE 5000

# 运行Flask应用
CMD ["python3", "app.py"]
