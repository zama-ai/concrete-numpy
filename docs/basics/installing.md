# Installation

**Concrete Numpy** is natively supported on Linux and macOS for Python 3.8 and 3.9, but if you have Docker support in your platform, you can use the docker image to use **Concrete Numpy**.

## Using PyPI

You can install **Concrete Numpy** from PyPI:

```shell
pip install concrete-numpy
```

{% hint style="warning" %}
Apple silicon users must use docker installation (explained below) as there is no ARM version of some of our dependencies for the time being.
{% endhint %}

You can install the extra python dependencies for drawing circuits:

```shell
pip install concrete-numpy[full]
```

{% hint style="info" %}
**Concrete Numpy** depends on `pygraphviz` for drawing, which requires `graphviz` packages to be installed on your system (see [pygraphviz installation documentation](https://pygraphviz.github.io/documentation/stable/install.html)).
{% endhint %}

## Using Docker

You can also get the **Concrete Numpy** docker image:

```shell
docker pull zamafhe/concrete-numpy:v0.7.0
```

### Starting Jupyter server

By default, the entry point of the **Concrete Numpy** docker image is a jupyter server that you can access from your browser:

```shell
docker run --rm -it -p 8888:8888 zamafhe/concrete-numpy:v0.7.0
```

To save notebooks on host, you can use a local volume:

```shell
docker run --rm -it -p 8888:8888 -v /path/to/notebooks:/data zamafhe/concrete-numpy:v0.7.0
```

### Starting Bash session

Alternatively, you can launch a Bash session:

```shell
docker run --rm -it zamafhe/concrete-numpy:latest /bin/bash
```
