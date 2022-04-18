FROM ubuntu:20.04

COPY . /Docker

WORKDIR /Docker

RUN apt-get update && apt-get install -y

#Python env setup
RUN apt install python3-pip -y
RUN apt install python3-autopep8 -y

RUN pip3 install -r requirements.txt
RUN pip3 install tensorflow==2.8.0 --no-cache-dir

CMD ["python3", "./main.py"]
