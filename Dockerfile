FROM python:3.8.9
RUN mkdir microservice
WORKDIR /microservice
ADD . .
RUN pip install -r requirements.txt -i https://pypi.mirrors.ustc.edu.cn/simple
ENTRYPOINT ["python", "-m", "bdilab_server"]