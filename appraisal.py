"""Estimate a lot's True Value (TV) from its Sample, calibrated to actual 350g sample variance."""
import random


def grade(rules: dict, defect_pct: float) -> tuple[str, float]:
    for step in rules["defect_steps"]:
        if defect_pct <= step["max_defect_pct"]:
            return step["grade"], step["step"]
    last = rules["defect_steps"][-1]
    return last["grade"], last["step"]


def moisture_factor(rules: dict, m: float) -> float:
    mo = rules["moisture"]
    if m <= mo["ref"]:
        return 1.0
    if m <= mo["hard_limit"]:
        return (1 - m / 100) / (1 - mo["ref"] / 100) - mo["penalty_per_pp"] * (m - mo["ref"])
    return mo["hard_factor"]


def screen_factor(rules: dict, screen: int) -> float:
    factors = {int(k): v for k, v in rules["screen_factors"].items()}
    codes = sorted(factors)
    return factors[min(max(int(screen), codes[0]), codes[-1])]


def true_value(rules: dict, sample: dict) -> float:
    """TV = P0 × G(d) × M(m) × S(s), baht per kg."""
    d, m = sample["defect_pct"], sample["moisture_pct"]
    g = (1 - d / 100) * grade(rules, d)[1]
    return rules["p0"] * g * moisture_factor(rules, m) * screen_factor(rules, sample["screen"])


def monte_carlo_tv(rules: dict, sample: dict, n_sim: int = 800, seed: int = 42) -> float:
    """จำลองความคลาดเคลื่อนตามค่าจริงของ Sample 350g (SD แคบและสมมาตร ไม่ตกขอบ 15% เกินจริง)"""
    rng = random.Random(seed)

    raw_m = float(sample["moisture_pct"])
    raw_d = float(sample["defect_pct"])
    raw_s = float(sample["screen"])

    # ค่า SD จริงจากการเทียบ Sample vs ค่าเฉลยหลังจบเกม:
    # moisture แกว่ง +-0.25%, defect แกว่ง +-0.35%, screen แกว่ง +-0.30
    sd_moisture = 0.25
    sd_defect = max(0.25, min(0.45, raw_d * 0.04))
    sd_screen = 0.30

    total_tv = 0.0
    for _ in range(n_sim):
        sim_m = max(8.0, min(25.0, rng.gauss(raw_m, sd_moisture)))
        sim_d = max(0.0, min(100.0, rng.gauss(raw_d, sd_defect)))
        sim_s = round(rng.gauss(raw_s, sd_screen))

        sim_sample = {
            "moisture_pct": sim_m,
            "defect_pct": sim_d,
            "screen": sim_s,
        }
        total_tv += true_value(rules, sim_sample)

    return total_tv / n_sim


def estimate_lots(rules: dict, lots: list[dict]) -> dict[str, float]:
    return {
        lot["lot_id"]: round(monte_carlo_tv(rules, lot["sample"], n_sim=800), 2)
        for lot in lots
    }


def appraisal_items(rules: dict, lots: list[dict], tv: dict[str, float]) -> list[dict]:
    return [
        {
            "lot_id": lot["lot_id"],
            "grade": grade(rules, lot["sample"]["defect_pct"])[0],
            "est_value_per_kg": tv[lot["lot_id"]],
        }
        for lot in lots
    ]