# xai

Research work extending the explainable AI approach of the van der Schaar lab. See:
Crabbé, J., Qian, Z., Imrie, F., & van der Schaar, M. (2021). Explaining latent representations with a corpus of examples. In Advances in Neural Information Processing Systems


## Background


## How to use


## Dev environment setup
```shell
conda create -n xai python=3.11
conda activate xai
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

Compile requirements
```shell
pip install pip-tools

pip-compile requirements.in -- upgrade
pip-compile requirements-dev.in -- upgrade
```

### Style and Testing
Testing is done using `pytest`.

Dosctrings conform to the [numpy docstring style guide](https://numpydoc.readthedocs.io/en/latest/format.html) 
and `flake8` is used to lint.
