# Example on propagation

We consider an example propagating the initial NRHO state in Moon-centered inertial frame.

We begin with package imports & furnishing SPICE kernels

```python
import numpy as np
import spiceypy as spice
import matplotlib
import matplotlib.pyplot as plt
import os

import paladin

spice.furnsh(os.path.join(os.getenv("SPICE"), "lsk", "naif0012.tls"))
spice.furnsh(os.path.join(os.getenv("SPICE"), "spk", "de440.bsp"))
spice.furnsh(os.path.join(os.getenv("SPICE"), "pck", "gm_de440.tpc"))

spice.furnsh(os.path.join(os.path.dirname(__file__), "..", "assets", "spice", "earth_moon_rotating_mc.tf"))  # custom frame kernel
spice.furnsh(os.path.join(
    os.path.dirname(__file__), "..",
    "assets",
    "spice",
    "receding_horiz_3189_1burnApo_DiffCorr_15yr.bsp"
))  # baseline NRHO
```

We now query the initial state

```python
# set epochs
et0 = spice.utc2et("2025-01-01T08:09:36")
tf = 60 * 86400.0  # seconds
ets = np.linspace(et0, et0 + tf, 3000)
naif_frame = "J2000"

# extract states from SPICE
sv_baseline = np.zeros((len(ets),6))
for idx,et in enumerate(ets):
    sv_sc,_ = spice.spkezr("-60000", et, naif_frame, "NONE", "399")
    sv_moon, _ = spice.spkezr("301", et, naif_frame, "NONE", "399")
    sv_baseline[idx,:] = sv_sc - sv_moon
```

We now make the propagator object

```python
# create NRHO propagator
naif_ids = ["301", "399", "10"]
mus = [spice.bodvrd(ID, "GM", 1)[1][0] for ID in naif_ids]
prop_nbody = paladin.GSLPropagatorNBody(
    naif_frame,
    naif_ids,
    mus,
    lstar = 3000.0,
    use_canonical=True,
)
prop_nbody.summary()

# solve IVP
x0 = prop_nbody.dim2nondim(sv_baseline[0,:])
tf_nondim = tf/prop_nbody.tstar
res_nbody = prop_nbody.solve(
    et0,
    [0, tf_nondim],
    x0,
    t_eval=np.linspace(0, tf_nondim, len(ets))
)
```

We can make a plot of the state history queried from the SPICE kernels and the propagated state:

```python
# make a plot
fig = plt.figure(figsize = (8,8))
ax = plt.axes(projection = '3d')
ax.plot(sv_baseline[:,0], sv_baseline[:,1], sv_baseline[:,2],
    color="black", label="Baseline")
paladin.set_equal_axis(ax, sv_baseline[:,0], sv_baseline[:,1], sv_baseline[:,2])

# append to plot
ax.plot(res_nbody.y[0,:]*prop_nbody.lstar,
        res_nbody.y[1,:]*prop_nbody.lstar,
        res_nbody.y[2,:]*prop_nbody.lstar,
    color="red", label="Propagated", linewidth=0.5)
ax.set(xlabel="x, km", ylabel="y, km", zlabel="z, km")
ax.legend()
plt.savefig("../plots/propagation_example_nrho.png", dpi=200)

# error plot
error_pos = res_nbody.y[0:3,:]*prop_nbody.lstar - np.transpose(sv_baseline[:,0:3])
fig1, axs1 = plt.subplots(3,1, figsize=(9,5))
for idx in range(3):
    axs1[idx].plot(ets, error_pos[idx,:])
    axs1[idx].grid(True, alpha=0.5)
axs1[0].set(xlabel="Epoch", ylabel="x error, km")
axs1[1].set(xlabel="Epoch", ylabel="y error, km")
axs1[2].set(xlabel="Epoch", ylabel="z error, km")
plt.tight_layout()
plt.show()
```