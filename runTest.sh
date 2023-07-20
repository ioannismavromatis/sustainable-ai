
#!/bin/bash

# Description: Run test experiments for the Sustainable AI project
# Usage: ./runTest.sh [experiment]
# Author: Ioannis Mavromatis

# Date: 20/07/2023

if [ $# -eq 0 ]
then
    echo "No argument supplied. Please provide one of the following arguments: tooltest, batchtest"
    exit 1
fi

ARGUMENT=$1
EPOCH=200

# Set the environment variables - make sure a venv created with poetry is available
# PATH=$PATH:/home/synergia/.local/bin
# PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring
# shopt -s expand_aliases
# alias pythonenv="~/.cache/pypoetry/virtualenvs/sustainable-ai-qHj6JKzP-py3.9/bin/python3"

declare models=(LeNet MobileNet EfficientNetB0 SimpleDLA ResNet18 VGG PreActResNet18 GoogLeNet DenseNet121 ResNeXt29_2x64d MobileNetV2 SENet18 RegNetX_200MF DPN26)
declare batch_sizes=(256 224 192 160 128 96 64 32 16 8 4)

if [ $ARGUMENT = "tooltest" ]; then
    echo "Running tools test"

    rm -rv ./results
    mkdir ./results
    sudo dmidecode -t memory > ./results/meminfo.txt

    for model in "${models[@]}"; do
        # Run experiments using our FROST tool
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
elif [ $ARGUMENT = "batchtest" ]; then
    echo "Running batch test"
    
    mkdir ./results_bk

    rm -rv ./results
    mkdir ./results
    sudo dmidecode -t memory > ./results/meminfo.txt

    for batch in "${batch_sizes[@]}"; do
        ls ./results/ | grep -xv "meminfo.txt" | xargs -I {} rm -r ./results/{}
        for model in "${models[@]}"; do    
            if [ "$model" != "GoogLeNet" ]; then
                python3 main.py --get-stats --epoch ${EPOCH} --network ${model} --batch-size ${batch} --test-size ${batch}
            fi
        done
        cp -rT ./results ./results_bk/results_${batch}
    done
else
    echo "Invalid argument. Please provide one of the following arguments: tooltest, batchtest"
    exit 1
fi
