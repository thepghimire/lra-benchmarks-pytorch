# lra-benchmark

This repository is based on the Long Range Arena (LRA) datasets from Google Research. The original repository by Google is available [here](https://github.com/google-research/long-range-arena). The paper is available on [ArXiv](https://arxiv.org/pdf/2011.04006.pdf).
In this repo we provide researchers code to work with some of the simpler tasks. It also has the advantage of using PyTorch and HuggingFace Transformers instead of Jax/Flax, with which less are familiar - thus more easily tweakable.

## Use Benchmark
1. Run `sh ./get_data.sh`. It will create a new `datasets` directory in the project and will populate it with the required datasets (make sure to run this in the project's root directory as the script uses relative paths)
2. To test the code with simple models, just run: `python run_model.py`
