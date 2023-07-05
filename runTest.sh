
#!/bin/bash

# Description: Run test experiments for the Sustainable AI project
# Usage: ./runTest.sh
# Author: Ioannis Mavromatis

# Date: 05/04/2023

# Set the environment variables - make sure a venv created with poetry is available
# PATH=$PATH:/home/synergia/.local/bin
# PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring
# shopt -s expand_aliases
# alias pythonenv="~/.cache/pypoetry/virtualenvs/sustainable-ai-qHj6JKzP-py3.9/bin/python3"

declare models=(LeNet MobileNet EfficientNetB0 SimpleDLA ResNet18 VGG PreActResNet18 GoogLeNet DenseNet121 ResNeXt29_2x64d MobileNetV2 SENet18 RegNetX_200MF DPN26)


EPOCH=1

rm -rv ./results
mkdir ./results
sudo dmidecode -t memory > ./results/meminfo.txt

for model in "${models[@]}"; do
    # Run experiments using our inhouse tool
    python3 main.py --get-stats --epoch ${EPOCH} --network ${model}

    sleep 2

    # Run experiments without any monitoring tool
    python3 main.py --epoch ${EPOCH} --network ${model}

    sleep 2

    # Run experiments using the eco2ai tool
    python3 main.py --tool eco2ai --epoch ${EPOCH} --network ${model}

    sleep 2

    # Run experiments using the codecarbon tool
    python3 main.py --tool codecarbon --epoch ${EPOCH} --network ${model}

    sleep 2

    # Run experiments using the carbontracker tool
    python3 main.py --tool carbontracker --epoch ${EPOCH} --network ${model}
done