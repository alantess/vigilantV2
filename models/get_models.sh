#! /bin/bash


echo "Gathering model"

wget https://github.com/alantess/vigilantV2/releases/download/v0.1-beta/lanes_segnet.pt
wget https://github.com/alantess/vigilantV2/releases/download/v0.1-beta/traced_lanesNet.pt


mv traced_lanesNet.pt quantized_lanesNet.pt
