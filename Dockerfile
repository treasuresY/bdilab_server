FROM python:3.8.9

RUN mkdir microservice
WORKDIR /microservice

# mesa-libGL: this is to avoid "ImportError: libGL.so.1" from opencv
RUN apt-get update && apt-get install -y libgl1-mesa-glx

#RUN apt-get update && \
#    apt-get install -y --no-install-recommends \
#        mesa-utils \
#        libgl1-mesa-glx \
#        libegl1-mesa \
#        libgl1-mesa-dri \
#        && \
#    rm -rf /var/lib/apt/lists/*

ADD . .
RUN pip install --no-cache-dir -r requirements.txt -i https://pypi.mirrors.ustc.edu.cn/simple
# 因为bdilab-detect项目上传到了Pypi ,清华镜像中没有,因此需要从Pypi拉取;
RUN pip install bdilab-detect==0.2.6.dev
ENTRYPOINT ["python", "-m", "bdilab_server"]