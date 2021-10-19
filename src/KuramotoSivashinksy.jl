module KuramotoSivashinksy

using FFTW

export ksintegrateNaive, ksintegrateTuned

function ksintegrateNaive(u, Lx, dt, Nt, nplot);
    Nx = length(u)                  # number of gridpoints
    kx = vcat(0:Nx/2-1, 0, -Nx/2+1:-1)  # integer wavenumbers: exp(2*pi*i*kx*x/L)
    alpha = 2*pi*kx/Lx              # real wavenumbers:    exp(i*alpha*x)
    D = 1im*alpha                   # D = d/dx operator in Fourier space
    L = alpha.^2 - alpha.^4         # linear operator -D^2 - D^4 in Fourier space
    G = -0.5*D                      # -1/2 D operator in Fourier space
    Nplot = round(Int64, Nt/nplot)+1  # total number of saved time steps
    
    x = collect(0:Nx-1)*Lx/Nx
    t = collect(0:Nplot)*dt*nplot
    U = zeros(Nplot, Nx)
    
    # some convenience variables
    dt2  = dt/2
    dt32 = 3*dt/2
    A =  ones(Nx) + dt2*L
    B = (ones(Nx) - dt2*L).^(-1)

    Nn  = G.*fft(u.*u); # -1/2 d/dx(u^2) = -u u_x, collocation calculation
    Nn1 = Nn;

    U[1,:] = u; # save initial value u to matrix U
    np = 2;     # counter for saved data
    
    # transform data to spectral coeffs 
    u  = fft(u);

    # timestepping loop
    for n = 0:Nt-1
        Nn1 = Nn;                       # shift N^{n-1} <- N^n
        Nn  = G.*fft(real(ifft(u)).^2); # compute N^n = -u u_x

        u = B .* (A .* u + dt32*Nn - dt2*Nn1); # CNAB2 formula
        
        if mod(n, nplot) == 0
            U[np,:] = real(ifft(u))
            np += 1            
        end
    end  
    U,x,t
end

function ksintegrateTuned(u, Lx, dt, Nt);
    u = (1+0im)*u                       # force u to be complex
    Nx = length(u)                      # number of gridpoints
    kx = vcat(0:Nx/2-1, 0:0, -Nx/2+1:-1)# integer wavenumbers: exp(2*pi*kx*x/L)
    alpha = 2*pi*kx/Lx                  # real wavenumbers:    exp(alpha*x)
    D = 1im*alpha                       # spectral D = d/dx operator 
    L = alpha.^2 - alpha.^4             # spectral L = -D^2 - D^4 operator
    G = -0.5*D                          # spectral -1/2 D operator, to eval -u u_x = 1/2 d/dx u^2

    # convenience variables
    dt2  = dt/2
    dt32 = 3*dt/2
    A =  ones(Nx) + dt2*L
    B = (ones(Nx) - dt2*L).^(-1)

    # compute in-place FFTW plans
    FFT! = plan_fft!(u, flags=FFTW.ESTIMATE)
    IFFT! = plan_ifft!(u, flags=FFTW.ESTIMATE)

    # compute uf == Fourier coeffs of u and Nnf == Fourier coeffs of -u u_x
    # FFT!(u);
    Nn  = G.*fft(u.^2); # Nnf == -1/2 d/dx (u^2) = -u u_x, spectral
    Nn1 = copy(Nn);     # use Nnf1 = Nnf at first time step
    FFT!*u;

    # timestepping loop, many vector ops unrolled to eliminate temporary vectors
    for n = 0:Nt

        for i = 1:length(Nn)
            @inbounds Nn1[i] = Nn[i];
            @inbounds Nn[i] = u[i];            
        end

        IFFT!*Nn; # in-place FFT

        for i = 1:length(Nn)
            @fastmath @inbounds Nn[i] = Nn[i]*Nn[i];
        end

        FFT!*Nn;

        for i = 1:length(Nn)
            @fastmath @inbounds Nn[i] = G[i]*Nn[i];
        end

        for i = 1:length(u)
            @fastmath @inbounds u[i] = B[i]* (A[i] * u[i] + dt32*Nn[i] - dt2*Nn1[i]);
        end

    end
    u = real(ifft(u))
end

end # module
