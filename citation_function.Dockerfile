FROM python:3.6

LABEL maintainer="1262758612@qq.com"

# citation function port
EXPOSE 38081

#RUN apt-get update && apt-get install -y --no-install-recommends \
#        software-properties-common  && \
#     apt-get clean

#RUN add-apt-repository ppa:jonathonf/python-3.6 && \
#        apt-get update && \
#        apt-get install -y --no-install-recommends \
#        curl \
#        python3.6 \
#        python3.6-dev && \
#    apt-get clean && \
#    rm -rf /var/lib/apt/lists/*

#RUN apt-get install -y --no-install-recommends curl python3-distutils
#RUN curl -O https://bootstrap.pypa.io/get-pip.py
#COPY get-pip.py get-pip.py
    # if install pip for python3.6, only need to change python -> python3.6
#RUN python3.6 get-pip.py && \
#    rm get-pip.py

RUN pip install flask numpy nltk keras==2.2.4 -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install requests -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install tensorflow==1.13.1 -i https://pypi.tuna.tsinghua.edu.cn/simple

WORKDIR /

COPY serving_client.py serving_client.py
COPY dict.json /dict.json
COPY nltk_data /root/nltk_data

ENTRYPOINT ["python", "serving_client.py"]

#/home/zhangli/notebook_dir/jupyter_notebook
#ENTRYPOINT ["/usr/bin/tf_serving_entrypoint.sh"]
