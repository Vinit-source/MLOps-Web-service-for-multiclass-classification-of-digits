sudo docker build --rm -f docker/Dockerfile . -t mnist:latest
sudo docker run -it mnist:latest