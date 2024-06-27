# Setting up `paladin`

## 1. Installation & Setup

### Setup GSL

See the [pygsl github page](https://github.com/pygsl/pygsl) for more details.

#### On Windows:

For example, follow: [https://github.com/hariseldon99/GSL-WIN64?tab=readme-ov-file](https://github.com/hariseldon99/GSL-WIN64?tab=readme-ov-file)


#### On Mac:

```
sudo port install gsl
```

#### On Linux:

```
sudo apt-get install libgsl-dev
```




### Install `paladin`

To install via pip

```bash
pip install spacecraft-paladin
```

To uninstall

```bash
pip uninstall spacecraft-paladin
```


### SPICE kernels

Users are responsible for downloading [the generic SPICE kernels froom the NAIF website](https://naif.jpl.nasa.gov/pub/naif/generic_kernels/). In addition, supplementary custom kernels specific to this module are stored in `paladin/assets/spice/`. The most commonly required kernels are:

- `naif0012.tls`
- `de440.bsp`
- `gm_de440.tpc` 

