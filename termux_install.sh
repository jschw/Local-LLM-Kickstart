#!/bin/bash
pkg update
pkg install -y python git termux-api python-pip cmake clang wget ninja patchelf build-essential matplotlib rust binutils libzmq python-onnxruntime

termux-setup-storage

mkdir -p ~/python-env/chatshell
python -m venv --system-site-packages ~/python-env/chatshell
source ./python-env/chatshell/bin/activate

pip install --verbose -r requirements.txt
