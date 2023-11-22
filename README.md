# xai
Explainable AI research repository.


## Background
This repository contains research work extending the explainable AI approach of the van der Schaar lab. 
See: `Crabbé, J., Qian, Z., Imrie, F., & van der Schaar, M. (2021). Explaining latent representations with a corpus of examples. In Advances in Neural Information Processing Systems`

Specifically, we seek to use the Simplex explainability approach to establish global confidence measures
to assess the appropriateness of a given model to a particular dataset.

Various experiments are run to explore the latent space and apply different distance measures based on Simplex
approximations. These are applied in two settings: MNIST handwritten digits and histopathological lung and colon
images to classify cancer.

All code use to handle data, train models, and run experiments is contained in this repo, 
along with saved models, results and figures.
these can be run with the environment set up described in the next section. 


## Dev environment setup
this section contains notes to install and run the models and experiments in this repo.

Create an environment.
```shell
conda create -n xai python=3.11
conda activate xai
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -r requirements.txt
pip install -r requirements-dev.txt
pip install simplexai
pip install -e .
```

Recompile requirements (only required if new dependencies are added or upgrading existing libraries).
```shell
pip install pip-tools

pip-compile requirements.in --upgrade
pip-compile requirements-dev.in --upgrade
```

### Style and Testing
Testing is done using `pytest`.

Docstrings conform to the [numpy docstring style guide](https://numpydoc.readthedocs.io/en/latest/format.html) 
and `flake8` is used to lint.
