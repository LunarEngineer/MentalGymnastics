docker build . -t mentalgymnastics
# Example of docker command (note that this is overload cschranz)
docker run --gpus all -d -it --rm -p 8888:8888 -v $(pwd):/home/jovyan/work -e GRANT_SUDO=yes -e JUPYTER_ENABLE_LAB=yes --name mentalgym_gpu --user root mentalgymnastics:latest