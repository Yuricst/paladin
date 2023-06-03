# luna2

Cislunar Astrodynamics in Simplified and Full Ephemeris models

<p align="center">
  <img src="./assets/Luna_II.png" width="250" title="luna2">
</p>

## Overview

This package provides tools for conducting CR3BP and ephemeris level analysis in cislunar (and R3BP) environments. 

While the library is implemented in python, the majority of functionalities are powered by either [SPICE routines](https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/FORTRAN/spicelib/index.html) or through numpy/scipy/numba implementations, resulting in (relatively) fast computations. 
Furthermore, no compiling is needed, making the library easily portable & ready to go.

Optimization is conducted by constructing problems as [`pygmo` udp's](https://esa.github.io/pygmo2/index.html), which can then be solved through a variety of compatible solvers, including IPOPT, SNOPT, or WORHP (the latter two requires licenses). 


## Dependencies

- `numpy`, `matplotlib`, `numba`, `scipy`, `spiceypy`, `pygmo`, `pygmo_plugins_nonfree`


## SPICE setup

Users are responsible for downloading [the generic SPICE kernels froom the NAIF website](https://naif.jpl.nasa.gov/pub/naif/generic_kernels/). In addition, supplementary custom kernels specific to this module are stored in `luna2/assets/spice/`. The most commonly required kernels are:

- `naif0012.tls`
- `de440.bsp`
- `gm_de440.tpc` 


## Installation

This package is still in development. For now, please clone the repository and add to path.


## Capabilities

#### Roadmap

- [x] Propagation in CR3BP
- [x] Propagation in N-body problem
- [ ] Transition to full-ephemeris model
- [x] Helper methods for frame transformation


## Gallery

NRHO propagation

<p align="center">
  <img src="./plots/propagation_example_nrho.png" width="400" title="Propagation example">
</p>



## On Luna II

<blockquote>
月の反対側のL3宙域に存在する地球連邦軍の宇宙要塞。菱形状の外観を有し、その全幅は180kmにも及ぶ月以外では地球圏最大の天体である[1]。
元はスペースコロニー建設用の資源衛星としてアステロイドベルトから地球圏に移送された小惑星ユノーであり、宇宙世紀0045年に月軌道に固定された。その後、0060年に「60年代軍備増強計画」の一環として連邦軍の軍事拠点として転用され、0070年にはサイド7建設の名目でL3に移動している。

月の反対側という立地は、ジオン公国はもとより、サイド7以外の各サイドからも離れた場所に位置している。これは各地の状況を俯瞰出来る事から敵の初撃による壊滅を避けつつ、各地の駐留部隊に指示を出すのに適している。各サイドへの移動は困難だが宇宙勢力の攻撃を受けにくく、また辺境故に機密保持にも適していた。

恒久軍事基地として機能する関係上、内部には様々な軍事施設や居住区画が存在し、表面にも宇宙港の出入口や監視システム、対空火器が存在するものの、そのサイズ故に監視網に穴が開きやすい。

兵器生産のための工廠も有しており、一年戦争～グリプス戦役にかけて様々な機体を開発・生産している[2]。

一年戦争時はその地勢が最大に活かされ、緒戦で各サイドの駐留部隊が壊滅した後もジオンの大規模攻勢を受けないまま連邦軍唯一の宇宙拠点として機能した。また、ガンダリウム合金の素材となる希少金属の最大鉱床でもあり、資源衛星としての価値も高い。
</blockquote>

[ルナツー（Luna 2）](https://gundam.wiki.cre.jp/wiki/%E3%83%AB%E3%83%8A%E3%83%84%E3%83%BC)

