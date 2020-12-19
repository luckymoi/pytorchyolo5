#!/bin/bash
DIR=`dirname $0`

echo "!!!IMPORTANT!!!"
echo "You need to install Pytorch first https://pytorch.org/get-started/locally/"

echo "Installing pytorch dependencies..."
sudo apt-get update
sudo apt-get install python3-numpy


if [ ! -e "./conf.json" ]; then
    echo "Creating conf.json"
    sudo cp conf.sample.json conf.json
else
    echo "conf.json already exists..."
fi
echo "Adding Random Plugin Key to Main Configuration"
node $DIR/../../tools/modifyConfigurationForPlugin.js pytorch key=$(head -c 64 < /dev/urandom | sha256sum | awk '{print substr($1,1,60)}')

