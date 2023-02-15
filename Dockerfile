FROM python:3.8.9
RUN mkdir microservice
WORKDIR /microservice
ADD . .
RUN pip install -r requirements.txt -i https://pypi.mirrors.ustc.edu.cn/simple
# 因为bdilab-detect项目上传到了Pypi ,清华镜像中没有,因此需要从Pypi拉取;
RUN pip install bdilab-detect==0.1.0.dev1
RUN rm ~/.cache/pip -rf
ENTRYPOINT ["python", "-m", "bdilab_server"]