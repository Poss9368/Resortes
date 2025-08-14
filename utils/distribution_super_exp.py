import math
import random
import bisect

class SamplerSuperExp:
    """
    P(n) = A / n^n, n=1,2,...
    Construye una CDF truncando cuando la masa restante es < tol, o hasta n_max si se indica.

    Parámetros:
      - tol: tolerancia para truncar la cola
      - n_max: máximo n (opcional)
      - seed: semilla para reproducibilidad (opcional)
    """
    def __init__(self, tol=1e-12, n_max=None, seed=None):
        self.rng = random.Random(seed)  # generador propio con semilla

        ws = []  # pesos no normalizados: w_n = n^(-n)
        n = 1
        while True:
            w = n ** (-n)
            ws.append(w)

            # Si hay tope explícito, seguimos hasta n_max
            if n_max is not None and n >= n_max:
                break

            # Razón r = w_{n+1}/w_n = (n/(n+1))^n * 1/(n+1)
            r = (n / (n + 1)) ** n / (n + 1)

            # Cota superior del resto tras agregar w: w * r / (1 - r)
            tail_upper = (w * r / (1 - r)) if r < 1 else float("inf")

            if tail_upper < tol:
                break

            n += 1

        s = sum(ws)
        self.A = 1.0 / s
        self.pmf = [self.A * w for w in ws]  # P(n) para n=1..N

        # CDF acumulada para búsqueda binaria
        c = 0.0
        self.cdf = []
        for p in self.pmf:
            c += p
            self.cdf.append(c)

    def sample(self) -> int:
        u = self.rng.random()  # usar el RNG con semilla
        idx = bisect.bisect_left(self.cdf, u)
        return idx + 1  # porque n arranca en 1

    def pmf_value(self, n: int) -> float:
        # Devuelve P(n) dentro del rango precomputado; fuera, aproxima con A/n^n
        if 1 <= n <= len(self.pmf):
            return self.pmf[n - 1]
        return self.A / (n ** n)


if __name__ == "__main__":
    sampler1 = SamplerSuperExp(tol=1e-14, seed=42)
    sampler2 = SamplerSuperExp(tol=1e-14, seed=42)  # mismo seed para comparar
    print("A ≈", sampler1.A)
    muestras1 = [sampler1.sample() for _ in range(10)]
    muestras2 = [sampler2.sample() for _ in range(10)]
    print(muestras1)
    print(muestras2)  # idénticas por la misma semilla
