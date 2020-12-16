#!/bin/bash
DIR=`dirname $0`
echo "Installing coral dependencies..."
echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
sudo apt-get update
sudo apt-get install libedgetpu1-max
sudo apt-get install libatlas-base-dev
echo "Coral dependencies installed."
echo "Getting coral object detection models..."
mkdir -p models
wget "https://github.com/google-coral/edgetpu/raw/master/test_data/ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite"
mv ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite models/
wget "https://dl.google.com/coral/canned_models/coco_labels.txt"
mv coco_labels.txt models/
echo "Models downloaded."


npm install yarn -g --unsafe-perm --force
npm install --unsafe-perm
if [ ! -e "./conf.json" ]; then
    echo "Creating conf.json"
    sudo cp conf.sample.json conf.json
else
    echo "conf.json already exists..."
fi
echo "Adding Random Plugin Key to Main Configuration"
node $DIR/../../tools/modifyConfigurationForPlugin.js tensorflow-coral key=$(head -c 64 < /dev/urandom | sha256sum | awk '{print substr($1,1,60)}')

echo "!!!IMPORTANT!!!"
echo "IF YOU DON'T HAVE INSTALLED CORAL DEPENDENCIES BEFORE, YOU NEED TO PLUG OUT AND THEN PLUG IN YOUR CORAL USB ACCELERATOR BEFORE USING THIS PLUGIN"
