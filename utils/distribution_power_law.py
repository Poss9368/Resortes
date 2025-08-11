import math
import random
import bisect

class SamplerPowerLaw:
    """
    P(n) = A / n^alpha, n=1,2,... con alpha > 1
    Construye una CDF truncando cuando la cola teórica (integral) < tol,
    o hasta n_max si se indica.

    Parámetros:
      - alpha: exponente > 1
      - tol: tolerancia para truncar la cola
      - n_max: máximo n (opcional; manda sobre tol si se provee)
      - seed: semilla para reproducibilidad (opcional)
    """
    def __init__(self, alpha: float, tol: float = 1e-12, n_max: int | None = None, seed: int | None = None):
        if not (alpha > 1.0):
            raise ValueError("alpha debe ser > 1 para que la serie converja.")
        self.alpha = alpha
        self.rng = random.Random(seed)

        # Elegimos N: si no hay n_max, paramos cuando la cola integral < tol
        # Cola exacta por cota integral: sum_{k>N} k^{-alpha} <= ∫_{N}^{∞} x^{-alpha} dx
        # = N^{1-alpha}/(alpha-1)
        ws = []
        n = 1
        while True:
            w = n ** (-alpha)
            ws.append(w)

            if n_max is not None and n >= n_max:
                break

            # cota de cola tras incluir n: resto <= (n+1)^{1-alpha} / (alpha-1)
            tail_upper = ((n + 1) ** (1.0 - alpha)) / (alpha - 1.0)
            if n_max is None and tail_upper < tol:
                break

            n += 1

        s = sum(ws)
        if s <= 0 or not math.isfinite(s):
            raise ValueError("Suma de pesos inválida.")
        self.A = 1.0 / s
        self.pmf = [self.A * w for w in ws]

        # CDF para búsqueda binaria
        c = 0.0
        self.cdf = []
        for p in self.pmf:
            c += p
            self.cdf.append(c)

    def sample(self) -> int:
        u = self.rng.random()
        idx = bisect.bisect_left(self.cdf, u)
        return idx + 1  # n arranca en 1

    def pmf_value(self, n: int) -> float:
        if n < 1:
            return 0.0
        # Dentro del rango tabulado devolvemos el valor tabulado;
        # fuera, una aproximación usando A/n^alpha (A calculada con truncación).
        if n <= len(self.pmf):
            return self.pmf[n - 1]
        return self.A * (n ** (-self.alpha))


if __name__ == "__main__":
    # Ejemplo de uso
    sampler1 = SamplerPowerLaw(alpha=2.0, tol=1e-12, seed=123)
    sampler2 = SamplerPowerLaw(alpha=2.0, tol=1e-12, seed=123)
    print("A ≈", sampler1.A)
    print([sampler1.sample() for _ in range(10)])
    print([sampler2.sample() for _ in range(10)])  # idéntico por la misma seed
