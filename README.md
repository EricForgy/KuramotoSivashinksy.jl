# KuramotoSivashinksy.jl

The original code was taken from:

- [Test case for PDEs: Kuramoto-Sivashinksy (KS)](https://online.kitp.ucsb.edu/online/transturb17/gibson/html/5-kuramoto-sivashinksy.html)

and updated to run on modern (stable) versions of Julia.

## Usage


```julia
using KuramotoSivashinksy

Lx = 128
Nx = 1024
dt = 1/16
nplot = 8
Nt = 1600

x = Lx*(0:Nx-1)/Nx
u = cos.(x) + 0.1*cos.(x/16).*(2*sin.(x/16).+1)

U,x,t = ksintegrateNaive(u, Lx, dt, Nt, nplot);
```