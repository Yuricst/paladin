# `pygsl` setup


## 0. Install gsl 

### On Windows:
For example, follow: https://github.com/hariseldon99/GSL-WIN64?tab=readme-ov-file


### On Mac:

```
sudo port install gsl
```

### On Linux:

```
sudo apt-get install libgsl-dev
```


## 1. Install `pygsl`

### On mac: 

```
pip install pygsl
```

### On Linux:

If `pip install pygsl` does not work:

Download source: https://github.com/pygsl/pygsl/releases, then run:

```
tar -xvzf pygsl-x.y.z.tar.gz
cd pygsl-x.y.z
python setup.py gsl_wrappers
python setup.py config
sudo python setup.py install
```

Note this requires `swig` and `gcc`. 


## 2. ODE examples

See:

`dev/pygsl_examples/odeiv.py`

Also:

https://github.com/pygsl/pygsl/blob/main/examples/odeiv.py