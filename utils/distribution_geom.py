import math
import random
import bisect

class SamplerGeom:
    """
    Distribución geométrica con soporte n=1,2,...
    PMF: P(n) = p * (1-p)^(n-1)

    Parámetros:
      - p: parámetro de la geométrica (0<p<1)
      - tol: tolerancia para truncar la cola, usando (1-p)^N < tol
      - n_max: máximo n (opcional); si se provee, manda sobre tol
      - seed: semilla para reproducibilidad (opcional)
    """
    def __init__(self, p=0.5, tol=1e-12, n_max=None, seed=None):
        if not (0.0 < p < 1.0):
            raise ValueError("p debe estar en (0,1).")
        self.p = p
        self.rng = random.Random(seed)

        # Determinar tamaño de la tabla
        if n_max is None:
            # Cola exacta de la geométrica tras N términos: (1-p)^N
            # Buscamos N tal que (1-p)^N < tol  ->  N > log(tol)/log(1-p)
            N = max(1, int(math.ceil(math.log(tol) / math.log(1.0 - p))))
        else:
            N = int(n_max)
            if N < 1:
                raise ValueError("n_max debe ser >= 1.")

        # Construcción de PMF y CDF tabuladas (normalizando por seguridad)
        ws = [p * (1.0 - p) ** (n - 1) for n in range(1, N + 1)]
        s = sum(ws)
        self.A = 1.0 / s  # ~1 si N grande o tol pequeño
        self.pmf = [self.A * w for w in ws]

        c = 0.0
        self.cdf = []
        for q in self.pmf:
            c += q
            self.cdf.append(c)

    def sample(self) -> int:
        """Muestreo por inversa tabulada (bisect sobre la CDF precomputada)."""
        u = self.rng.random()
        idx = bisect.bisect_left(self.cdf, u)
        return idx + 1  # n arranca en 1

    def pmf_value(self, n: int) -> float:
        """PMF exacta (no tabulada): p (1-p)^(n-1)."""
        if n < 1:
            return 0.0
        return self.p * (1.0 - self.p) ** (n - 1)

    # (Opcional) muestreo por inversa exacta, sin tabla:
    def sample_inv_exact(self) -> int:
        """
        Usando F(n) = 1 - (1-p)^n.
        Si u ~ U(0,1), el menor n con u <= F(n) es:
            n = ceil( log(1-u) / log(1-p) )
        """
        u = self.rng.random()
        return max(1, int(math.ceil(math.log(1.0 - u) / math.log(1.0 - self.p))))


if __name__ == "__main__":
    sampler1 = SamplerGeom(p=0.3, tol=1e-12, seed=43)
    sampler2 = SamplerGeom(p=0.3, tol=1e-12, seed=43)
    print("A ≈", sampler1.A)
    print([sampler1.sample() for _ in range(10)])
    print([sampler2.sample() for _ in range(10)])  # idéntico por la misma seed
