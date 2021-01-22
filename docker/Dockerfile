ARG PYTORCH="1.6.0"
ARG CUDA="10.1"
ARG CUDNN="7"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"

RUN apt-get update && apt-get install -y ffmpeg libsm6 libxext6 git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install MMCV
RUN pip install mmcv-full==latest+torch1.6.0+cu101 -f https://openmmlab.oss-accelerate.aliyuncs.com/mmcv/dist/index.html

# Vim is important
RUN apt-get update && apt-get install -y vim curl

# Install Jupyter Notebook and NBBlack
RUN pip install jupyterlab nb-black

RUN curl -sL https://deb.nodesource.com/setup_12.x | bash - && apt-get install -y nodejs
RUN jupyter labextension install @axlair/jupyterlab_vim @arbennett/base16-nord

# Install TIMM
RUN pip install timm

# Install MMDetection dependencies
RUN conda clean --all
WORKDIR /opt/mmdetextended
COPY . .
ENV FORCE_CUDA="1"
RUN pip install -r requirements/build.txt
RUN pip install --no-cache-dir -e .
