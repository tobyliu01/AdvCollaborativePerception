from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .types import PSM


def _clip01(value: float) -> float:
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


@dataclass(slots=True)
class BetaTrustState:
    """
    Beta(alpha, beta) trust state over [0, 1].

    Mean: alpha / (alpha + beta)
    Variance: alpha*beta / ((alpha+beta)^2 * (alpha+beta+1))
    """

    alpha: float
    beta: float
    prior_alpha: float
    prior_beta: float
    min_param: float = 1e-6

    @property
    def mean(self) -> float:
        denom = self.alpha + self.beta
        if denom <= 0.0:
            return 0.5
        return self.alpha / denom

    @property
    def variance(self) -> float:
        s = self.alpha + self.beta
        if s <= 0.0:
            return 0.25
        return (self.alpha * self.beta) / (s * s * (s + 1.0))

    def propagate_prior_interpolation(self, omega: float) -> None:
        omega = _clip01(float(omega))
        self.alpha = (1.0 - omega) * self.alpha + omega * self.prior_alpha
        self.beta = (1.0 - omega) * self.beta + omega * self.prior_beta
        self._ensure_valid()

    def propagate_expectation(self, delta_mu: float, target_mu: float = 0.5) -> None:
        if delta_mu <= 0.0:
            return
        mu = self.mean
        precision = max(self.alpha + self.beta, self.min_param)
        mu = mu + (target_mu - mu) / delta_mu
        mu = _clip01(mu)
        self.alpha = mu * precision
        self.beta = (1.0 - mu) * precision
        self._ensure_valid()

    def propagate_precision(self, delta_nu: float, target_nu: float) -> None:
        if delta_nu <= 0.0:
            return
        mu = self.mean
        nu = self.alpha + self.beta
        nu = nu + (target_nu - nu) / delta_nu
        nu = max(nu, self.min_param)
        self.alpha = mu * nu
        self.beta = (1.0 - mu) * nu
        self._ensure_valid()

    def update_from_psms(
        self,
        psms: Iterable[PSM],
        negativity_bias: float = 1.0,
        negativity_threshold: float = 0.0,
    ) -> None:
        delta_alpha = 0.0
        delta_beta = 0.0
        negativity_bias = max(1.0, float(negativity_bias))
        negativity_threshold = _clip01(float(negativity_threshold))

        for psm in psms:
            value = _clip01(float(psm.value))
            confidence = _clip01(float(psm.confidence))
            neg_weight = negativity_bias if value < negativity_threshold else 1.0
            delta_alpha += confidence * value
            delta_beta += neg_weight * confidence * (1.0 - value)

        self.alpha += delta_alpha
        self.beta += delta_beta
        self._ensure_valid()

    def _ensure_valid(self) -> None:
        self.alpha = max(float(self.alpha), self.min_param)
        self.beta = max(float(self.beta), self.min_param)
