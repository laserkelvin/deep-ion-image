# Deep Ion Image

In this project we develop and test a suite of light-weight, deep convolutional neural network architectures for ion image reconstruction.

The core features we are interested in are:

1. De-noising autoencoders
2. Probabilistic image reconstruction
3. Representation learning

The first and second aspect tie into our primary use case: taking noisy ion images, and reconstructing the central component of the ion sphere projection, which is subsequently used in image analyses. The probabilistic aspect is particularly important, as it is an often overlooked part of an otherwise highly quantitative experimental method: reconstruction algorithms that are typically have no way to properly account for uncertainty arising from noisy, out of focus, or uncentered images. For the last part, we take a deeper dive into the layer/filter basis to identify how and what is being learned through training.

What is provided in this repository is a means to reproduce the work. In the near future, we are aiming to release a pre-trained set of weights and package for use in "real" analysis workflows. Primarily, this is because the image sizes that are used (128 x 128) are not really suited for practical use just yet, and this repository serves to demonstrate and test these concepts first.

## Workflow

For details on how the folder structure is set out, and where to look for code, refer to [Project Organization](#project-organization).

### Data

While the original training data is not included in this repository, we plan to upload our dataset to Zenodo. That said, the codebase included here includes the necessary script and code to regenerate the dataset locally. See the script `data/raw/make_images.py`.

The scripts will generate arrays and store them in HDF5 format, uncompressed and double precision. The routines generate a specified number of ion images, storing their `beta` and `center` values for subsequent tasks. The ion images are stored as `true` and `projection`; the former as the central distribution, and the latter the projected ion image (observed in experiments) generated using the forward Abel transform implemented in `PyAbel`. Finally, a `train/dev/test` split is done. The data augmentation pipeline (i.e. images retrieved at every training iteration) includes generating composite images from the pre-computed library of images, and adding either Gaussian or Poisson noise to the projected "experimental" image. The two types of noise constitute signal-independent (e.g. thermal) and signal-dependent (photon noise) sources of noise respectively; realistically you would want a mixture, but for simplicity we opted to just have one or the other chosen at random, and we wrap the functionality of `skimage` for the noise addition.

### Models

We have implemented a hierarchy of CNNs, as well as inherited code from other projects such as from the PyTorch Lightning Bolts. In general, models are written with PyTorch Lightning while layers are base `torch.nn` modules. 


## Project Organization
```

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   │                    predictions
    │   ├── figures        <- Generated graphics and figures to be used in reporting
    │   ├── presentation   <- Generated graphics and figures to be used in talks
    │   └── writeup        <- Write reports in Markdown and process with Pandoc
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   │── visualization  <- Scripts to create exploratory and results oriented visualizations
    │   │   └── visualize.py
    │   │
    │   └── pipeline       <- Scripts that automate the process of preparing,
    │       │                 cleaning, and formatting the project data. This
    │       │                 kind of script is not suited for a notebook, since
    │       │                 it should be run headlessly on a cluster of sorts.
    │       │                 
    │       │                 
    │       ├── main.py    <- This drives the entire pipeline in one script.
    │       ├── make_dataset.py
    │       ├── combine_dataset.py
    │       ├── clean_dataset.py
    │       └── augment_dataset.py
    │
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.testrun.org
```

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
This version of the cookiecutter template is modified by Kelvin Lee.
