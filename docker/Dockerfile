FROM ubuntu:18.04
COPY mnist_example /exp/mnist_example
COPY api /exp/api
COPY client /exp/client
COPY requirements.txt /exp/requirements.txt
RUN apt-get update && apt-get install -y python3 python3-pip
RUN pip3 install --no-cache-dir -r /exp/requirements.txt
# RUN mkdir /exp/mnist_example/models
WORKDIR /exp
EXPOSE 5000

CMD ["python3", "./mnist_example/main.py"]
# CMD ["python3", "./api/app.py"]
CMD ["export", "FLASK_APP=./api/app.py"]
CMD ["flask", "run", "--host=0.0.0.0"]

