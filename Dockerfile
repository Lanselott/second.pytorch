# From https://github.com/ufoym/deepo/blob/master/docker/Dockerfile.pytorch-py36-cu90

# ==================================================================
# module list
# ------------------------------------------------------------------
# python        3.6    (apt)
# pytorch       latest (pip)
# ==================================================================

FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04
RUN APT_INSTALL="apt-get install -y --no-install-recommends" && \
    PIP_INSTALL="python -m pip --no-cache-dir install --upgrade" && \
    GIT_CLONE="git clone --depth 10" && \
    rm -rf /var/lib/apt/lists/* \
           /etc/apt/sources.list.d/cuda.list \
           /etc/apt/sources.list.d/nvidia-ml.list && \
    apt-get update && \


# ==================================================================
# tools
# ------------------------------------------------------------------
    DEBIAN_FRONTEND=noninteractive $APT_INSTALL \
        build-essential \
        ca-certificates \
        wget \
        git \
        vim \
        fish \
        libsparsehash-dev \
        && \


# ==================================================================
# python
# ------------------------------------------------------------------
    DEBIAN_FRONTEND=noninteractive $APT_INSTALL \
        software-properties-common \
        && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive $APT_INSTALL \
        python3.6 \
        python3.6-dev \
        && \
    DEBIAN_FRONTEND=noninteractive $APT_INSTALL \    
        libsm6 \
        libxext6 \
        libxrender1 \
        && \
    wget -O ~/get-pip.py \
        https://bootstrap.pypa.io/get-pip.py && \
    python3.6 ~/get-pip.py && \
    ln -s /usr/bin/python3.6 /usr/local/bin/python3 && \
    ln -s /usr/bin/python3.6 /usr/local/bin/python && \
    $PIP_INSTALL \
        setuptools \
        && \
    $PIP_INSTALL \
        numpy \
        scipy \
        matplotlib \
        opencv-python \
        Cython \
        psutil \
        seaborn \ 
        ipython \
        && \
# ==================================================================
# pytorch
# ------------------------------------------------------------------
    $PIP_INSTALL \
        torch -f \
        # https://download.pytorch.org/whl/nightly/cu90/torch_nightly.html \
        https://download.pytorch.org/whl/torch_stable.html\
        && \
    $PIP_INSTALL \
        torchvision \
        spatial_correlation_sampler \ 
        && \
# ==================================================================
# config & cleanup
# ------------------------------------------------------------------
    ldconfig && \
    apt-get clean && \
    apt-get autoremove && \
    rm -rf /var/lib/apt/lists/* /tmp/* ~/*

RUN PIP_INSTALL="python -m pip --no-cache-dir install --upgrade" && \
    $PIP_INSTALL \
        shapely fire pybind11 tensorboardX protobuf \
        scikit-image numba pillow

WORKDIR /tmp/cmake
RUN wget https://cmake.org/files/v3.14/cmake-3.14.0.tar.gz && \
    tar -xzvf cmake-3.14.0.tar.gz > /dev/null

WORKDIR cmake-3.14.0
RUN ./bootstrap > /dev/null && \
    make -j$(nproc --all) > /dev/null && \
    make install > /dev/null

WORKDIR /
RUN rm -rf /tmp/cmake

WORKDIR /root
RUN wget https://dl.bintray.com/boostorg/release/1.68.0/source/boost_1_68_0.tar.gz
RUN tar xzvf boost_1_68_0.tar.gz
RUN cp -r ./boost_1_68_0/boost /usr/include
RUN rm -rf ./boost_1_68_0
RUN rm -rf ./boost_1_68_0.tar.gz
RUN git clone https://github.com/Lanselott/second.pytorch.git --depth 10
# RUN git clone https://github.com/traveller59/SparseConvNet.git --depth 10
# RUN cd ./SparseConvNet && python setup.py install && cd .. && rm -rf SparseConvNet
RUN git clone https://github.com/traveller59/spconv.git --recursive
RUN git clone https://github.com/NVIDIA/flownet2-pytorch.git && cd flownet2-pytorch && bash install.sh
# RUN cd ./spconv && python setup.py bdist_wheel &&  cd ./dist && pip install *.whl
ENV NUMBAPRO_CUDA_DRIVER=/usr/lib/x86_64-linux-gnu/libcuda.so
ENV NUMBAPRO_NVVM=/usr/local/cuda/nvvm/lib64/libnvvm.so
ENV NUMBAPRO_LIBDEVICE=/usr/local/cuda/nvvm/libdevice
ENV PYTHONPATH=/root/second.pytorch
ENV PYTHONPATH=$PYTHONPATH:/root/second.pytorch/second
ENV PYTHONPATH=$PYTHONPATH:/root/flownet2-pytorch/networks/resample2d_package

VOLUME ["/root/data"]
VOLUME ["/root/model"]
WORKDIR /root/second.pytorch/second
RUN ln -s ../../../lansechen-intern/dataset/kitti_tracking/ ./ 
ENTRYPOINT ["fish"]
