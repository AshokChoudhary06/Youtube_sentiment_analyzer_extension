#!/bin/bash
# Log everything to start_docker.log
exec > /home/ubuntu/start_docker.log 2>&1

echo "Logging in to ECR..."
aws ecr get-login-password --region eu-north-1 | docker login --username AWS --password-stdin 577638368185.dkr.ecr.eu-north-1.amazonaws.com

echo "Pulling Docker image..."
docker pull 577638368185.dkr.ecr.eu-north-1.amazonaws.com/yt-chrome-extension-1:v1

echo "Checking for existing container..."
if [ "$(docker ps -q -f name=ask-app)" ]; then
echo "Stopping existing container..."
docker stop ask-app
fi
if [ "$(docker ps -aq -f name=ask-app)" ]; then
echo "Removing existing container..."
docker rm ask-app
fi

echo "Starting new container..."
docker run -d -p 80:8000 --name ask-app 577638368185.dkr.ecr.eu-north-1.amazonaws.com/yt-chrome-extension-1:v1
echo "Container started successfully."