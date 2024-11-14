# Tree Species Classification in Brandenburg

## Installation

> [!IMPORTANT]
> If your hardware supports CUDA tensors, you need to (1) install the CUDA framework on your device and
> (2) update the `pyproject.toml` file such that torch wheels are installed with CUDA support.

The project uses [Poetry](https://python-poetry.org/) to mange dependencies and virtual environments. While you can use whatever tool you want, setup is easiest when using Poetry as well. After successfully installing Poetry, create a new virtual environment and install all dependencies by executing the command below from the project's root. You may need to select the correect python environment inside your IDE, when using one, afterwards.

```bash
poetry install
```

## Setup

As the project tries to be mostly self-contained, all raw, interim and final data products are stored within this project.

### Project Structure

The project expects the structure outlined below. Initial image chips should be placed into `data/raw`. The directory `trees_brandenburg` contains various processing functions used in the notebook(s).

```raw
trees-brandenburg
├───data
│   ├───interim
│   ├───processed
│   │   └───imgs
│   │       ├───GBI
│   │       ├───GKI
│   │       └───...
│   └───raw
│       ├───AB
│       ├───AHS
│       ├───ALB
│       └───...
├───licenses
├───models
├───notebooks
├───reports
├───tests
└───trees_brandenburg
    ├───external
    └───modelling
```

## License

The noteboook(s) and examples are licensed under the [CC-BY-4.0](./licenses/cc-by-4) license. The code is licensed under the [3 clause BSD](./licenses/3-clause-bsd) license.

## Acknowledgements

Sasank Chilamkurthy for providing a [tutorial](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html#finetuning-the-convnet) and code examples for transfer learning with PyTorch.
