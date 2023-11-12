env:
  conda create -n xai python=3.11
  conda activate xai
  conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
  pip install -r requirements.txt
  pip install -r requirements-dev.txt

updaterequirements:
  pip install pip-tools
  pip-compile requirements.in
  pip-compile requirements-dev.in