FROM ubuntu:22.04

RUN apt-get update && \
      apt-get -y install sudo

RUN apt-get -y install software-properties-common

RUN add-apt-repository --yes ppa:mosquitto-dev/mosquitto-ppa

RUN add-apt-repository --yes ppa:deadsnakes/ppa

RUN apt update

RUN apt -y install python3.10

RUN apt-get update

RUN apt update

RUN apt -y install python3-pip

RUN python3.10 -m pip install --upgrade pip

RUN pip install flask

RUN pip install paho-mqtt

RUN pip install nano

RUN pip install torch torchvision torchaudio

RUN pip install scipy

RUN useradd -m docker && echo "docker:docker" | chpasswd && adduser docker sudo

USER docker

CMD /bin/bash
