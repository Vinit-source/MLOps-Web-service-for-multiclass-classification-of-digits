sudo docker build --rm -f docker/Dockerfile . -t mnist:latest
sudo docker run -it -p 5000:5000 --rm -e LC_ALL=C.UTF-8 -e LANG=C.UTF-8 -e FLASK_APP=/exp/api/app.py  mnist:latest