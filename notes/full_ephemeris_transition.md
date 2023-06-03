# Transitioning to full-ephemeris model

## UDP formulation

Consider a trajectory with $N+1$ nodes ($N+1 \geq 2$), comprising of $N$ legs. 

Decision vector:

$$
X = \left[ \mathrm{et}_0, \boldsymbol{x}_0(0), \cdots, \boldsymbol{x}_N(0), \Delta t_0, \cdots, \Delta t_{N-1} \right]
\in \mathbb{R}^{8N}
$$

Residual vector (equality constraint):

$$
F = \begin{bmatrix}
    \boldsymbol{x}_1(-\Delta t_0/2) - \boldsymbol{x}_0(\Delta t_0/2)
    \\
    \boldsymbol{x}_2(-\Delta t_1/2) - \boldsymbol{x}_1(\Delta t_1/2) 
    \\
    \vdots
    \\
    \boldsymbol{x}_N(-\Delta t_{N-1}/2) - \boldsymbol{x}_{N-1}(\Delta t_{N-1}/2) 
\end{bmatrix}
\in \mathbb{R}^{6N}
$$

Jacobian matrix

$$
DF/DX
= 
\begin{bmatrix}
    A & B & C
\end{bmatrix}
$$

where 

$$
\begin{aligned}
    A &= \dfrac{\partial F}{\partial \mathrm{et}_0}
    = \begin{bmatrix}
        \dfrac{\partial \boldsymbol{x}_1(-\Delta t_0/2)}{\partial t}
        -
        \dfrac{\partial \boldsymbol{x}_0(\Delta t_0/2)}{\partial t}
        \\
        \vdots
        \\
        \dfrac{\partial \boldsymbol{x}_N(-\Delta t_{N-1}/2)}{\partial t}
        -
        \dfrac{\partial \boldsymbol{x}_{N-1}(\Delta t_{N-1}/2)}{\partial t}
    \end{bmatrix}
    \\[3.0em]
    B &= \begin{bmatrix}
        -\Phi_0 & \Phi_1 & & & 
        \\
         & -\Phi_1 & \Phi_2 & &
        \\
        & & \ddots & \ddots & 
        \\
         & & & -\Phi_{N-1} & \Phi_{N}
    \end{bmatrix}
    \\[3.0em]
    C &= \begin{bmatrix}
        -\dfrac{1}{2}\dot{x}_1(-\Delta t_0/2) - \dfrac{1}{2}\dot{x}_0( \Delta t_0/2)
        & & & 
        \\[1.0em]
         & 
         -\dfrac{1}{2}\dot{x}_2(-\Delta t_2/2) - \dfrac{1}{2}\dot{x}_1(\Delta t_1/2)
          & &
        \\[1.0em]
        & & \ddots  & 
        \\[1.0em]
         & & &
         -\dfrac{1}{2}\dot{x}_N(-\Delta t_{N-1}/2) - \dfrac{1}{2}\dot{x}_{N-1}(\Delta t_{N-1}/2)
         % \dfrac{\partial \boldsymbol{x}_N(\Delta t_{N-1}/2)}{\partial t} - \dfrac{\partial \boldsymbol{x}_{N-1}(-\Delta t_{N-1}/2)}{\partial t}
    \end{bmatrix}
\end{aligned}
$$
