FROM tf2_plus_qianlai:1.0.0-r1

ADD . /src
WORKDIR /src

EXPOSE 5000

RUN apt-get update
RUN apt-get install vim -y
RUN pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/

CMD ["/bin/bash"]
