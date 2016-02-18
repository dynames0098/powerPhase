from numpy import *
import os


# functions
def generate_signal(shape, type="real"):  # real, complex, 1d, 2d, nonnegative
    if type == "real":
        if shape.size == 1:
            return random.randn(shape)
        elif shape.size == 2:
            return random.randn(shape[0], shape[1])
    elif type == "complex":
        if shape.size == 1:
            return (random.randn(shape) + random.randn(shape) * 1j)
        elif shape.size == 2:
            return (random.randn(shape[0], shape[1]) + random.randn(shape[0], shape[1]) * 1j)
    else:
        print "generate_signal failed"


def generate_measurements(signal, numMeasure,
                          type="complex_gaussian"):  # Frourier, real gaussian, complex gaussian, coded Frourier
    if type == "complex_gaussian":
        A = random.randn(numMeasure, signal.size) + 1j * random.randn(numMeasure, signal.size)
        A /= sqrt(2)
        b = abs(matmul(A, signal.flatten())) ** 2
        return (A, b)
    elif type == "real_gaussian":
        A = random.randn(numMeasure, signal.size)
        b = abs(matmul(A, signal.flatten())) ** 2
        return (A, b)
    elif type == "fourier1d":
        b = abs(fft.fft(hstack((signal, zeros(numMeasure - signal.size))))) ** 2
        A = fft.fft(eye(numMeasure))
        return (A, b)
    elif type == "fourier2d":
        signal = vstack((hstack((signal, zeros([signal.shape[0], numMeasure[1] - signal.shape[1]]))),
                         zeros([numMeasure[0] - signal.shape[0], numMeasure[1]])))

        def A(signal):
            return fft.fft2(signal)

        b = abs(A(signal)) ** 2
        return (A, b)
    elif type == "coded_fourier1d":  # numMeasure[0]:number of blocks; numMeasure[1]:number of oversampling
        signal = hstack((signal, zeros(numMeasure[1] - signal.size)))
        L = numMeasure[0]
        r = random.randint(1, 5, L * numMeasure[1])
        d = zeros(numMeasure[1] * L, dtype="complex")
        d[r == 1] = 1j
        d[r == 2] = 1
        d[r == 3] = -1j
        d[r == 4] = -1
        r = random.rand(L * numMeasure[1])
        d[r > 0.2] = d[r > 0.2] * sqrt(3)
        d[r <= 0.2] = d[r <= 0.2] / sqrt(2)
        A = tile(fft.fft(eye(numMeasure[1])), [L, 1])
        for i in arange(L):
            A[i * numMeasure[1]:(i + 1) * numMeasure[1]] *= conj(d[i * numMeasure[1]:(i + 1) * numMeasure[1]])
        b = abs(matmul(A, signal)) ** 2
        return (A, b)
    elif type == "coded_fourier2d":  # numMeasure[0]:number of blocks numMeasure[1,2]:num of oversampling
        signal = vstack((hstack((signal, zeros([signal.shape[0], numMeasure[2] - signal.shape[1]]))),
                         zeros([numMeasure[1] - signal.shape[0], numMeasure[2]])))
        r = random.randint(1, 5, [numMeasure[0], numMeasure[1], numMeasure[2]])
        d = zeros([numMeasure[0], numMeasure[1], numMeasure[2]], dtype="complex")
        d[r == 1] = 1j
        d[r == 2] = 1
        d[r == 3] = -1j
        d[r == 4] = -1
        r = random.rand(numMeasure[0], numMeasure[1], numMeasure[2])
        d[r > 0.2] = d[r > 0.2] * sqrt(3)
        d[r <= 0.2] = d[r <= 0.2] / sqrt(2)

        def A(signal, mask=d, m=numMeasure):
            return fft.fft2(signal * conj(mask))

        b = abs(A(signal)) ** 2
        return (A, b)


def solver_errorReduction(A, b, maxiter=2500, relerror=1e-7, type="matirx"):
    z = random.randn(A.shape[1]) + random.randn(A.shape[1]) * 1j
    y = sqrt(b)
    for i in arange(maxiter):
        ls_b = matmul(A, z) / abs(matmul(A, z)) * y
        z = linalg.lstsq(A, ls_b)[0]
        r = mean(abs(abs(matmul(A, z)) ** 2 - b))
        if r < relerror:
            return (z, r)
    return (z, r)

def initial(A,b,sampling="eigen",maxiter=50,type="matrix"):
    m,n=A.shape
    z=random.randn(n)
    if sampling=="eigen":
        for i in arange(maxiter):
            z=matmul(A.conj().T,b*matmul(A,z))
            z=z/linalg.norm(z)
    elif sampling=="random":
        z=z/linalg.norm(z)
    return z*sqrt(n*sum(b)/sum(abs(A)**2))

def solver_wirtingerFlow(A, b, ini="eigen", maxiter=2500, tol=1e-7, type="matrix", tau=330):
    if ini=="random":
        z = random.randn(A.shape[1]) + random.randn(A.shape[1]) * 1j
    elif ini=="eigen":
        z = initial(A,b)
        norm_z2=sum(abs(z)**2)

    def mu(t, tau0=tau):
        return min(1 - exp(-t / tau0), 0.2)

    for i in arange(maxiter):
        Az = matmul(A, z)
        r = linalg.norm(abs(Az) ** 2 - b) / sqrt(A.shape[0])
        if r < tol:
            return (z, r)
        else:
            grad = 1.0 / A.shape[0] * matmul(A.conj().T, (abs(Az) ** 2 - b) * Az)
            z=z-mu(i)/norm_z2*grad
    return (z, r)

def residual(A,b,z):
    Az=matmul(A,z)
    return linalg.norm(abs(Az)**2-b)/A.shape[0]
def solver_simpleKaczmarz(A, b, sampling="cyclic", ini="eigen",maxiter=500,tol=1e-7,type="matrix"):
    m,n=A.shape
    y=sqrt(b)
    z=initial(A,b,ini)
    if sampling == "cyclic":
        for i1 in arange(maxiter):
            r=residual(A,b,z)
            print r
            if r<tol:
                return (z,r)
            for i2 in arange(m):
                angle=sum(A[i2]*z)
                angle=angle/abs(angle)
                z=z+(y[i2]*angle-sum(A[i2]*z))/sum(abs(A[i2])**2)*A[i2].conj()
        return (z,r)

def solver_blockKaczmarz(A, b):
    return 0


# scripts
s = generate_signal(array([10]), type="complex")
(A, b) = generate_measurements(s, array([60]),type="complex_gaussian")
(z, r1) = solver_wirtingerFlow(A, b)
(z, r2) = solver_errorReduction(A, b)
(z, r3) = solver_simpleKaczmarz(A, b)
print r1,r2,r3
# plots
