FROM continuumio/miniconda3

WORKDIR /home/app

RUN apt-get update
RUN apt-get install nano unzip
RUN apt install curl -y
RUN pip install pip pandas streamlit sklearn matplotlib numpy tensorflow Pillow pickle-mixin opencv-python mahotas
RUN apt-get install ffmpeg libsm6 libxext6 -y
RUN apt-get update

COPY . /home/app

CMD streamlit run --server.port $PORT main.py