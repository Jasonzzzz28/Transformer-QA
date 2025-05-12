FROM continuumio/miniconda3

# 复制环境文件
COPY environment.yml /tmp/environment.yml

# 创建 conda 环境
RUN conda env create -f /tmp/environment.yml

# 使 RUN 命令使用新环境
SHELL ["conda", "run", "-n", "transformerQA", "/bin/bash", "-c"]

# 设置默认环境
ENV PATH /opt/conda/envs/transformerQA/bin:$PATH
