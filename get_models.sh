#! /bin/bash


echo "Gathering model"

mkdir -p models
cd models
wget https://github.com/alantess/vigilantV2/releases/download/v0.1-beta/lanesNet.onnx
wget https://github.com/alantess/vigilantV2/releases/download/v0.1-beta/lanes_segnet.pt
wget https://github.com/alantess/vigilantV2/releases/download/v0.1-beta/quantized_lanesNet.pt

cd ..

echo "Gathering Video..."
cd etc
mkdir -p videos && cd videos
wget https://github.com/alantess/vigilantV2/releases/download/v0.1-beta/driving.mp4

cd ../../

echo "COMPLETED."
