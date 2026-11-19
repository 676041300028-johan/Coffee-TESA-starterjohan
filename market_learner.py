"""Online Market Clearing Model, Practice Warm-Start, and Monte Carlo Lookahead Evaluator."""
import json
import math
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np

from coffee_math import CoffeeLot, BuyerRoute
from history_tracker import classify_segment


class OnlineMarketClearingModel:
    """
    เรียนรู้เส้นโค้งโอกาสชนะ P(win | price_ratio, segment) แบบ Online
    จากทั้งผลแพ้-ชนะของทีมเรา (my_offers) และราคาเคลียร์จริงของตลาด (market_summary)
    """
    PRIOR_PARAMS = {
        "SPECIALTY": (1.19, 45.0),
        "COMMERCIAL": (1.012, 55.0),
        "INSTANT": (0.920, 55.0),
    }

    SEG_BOUNDS = {
        "SPECIALTY": (1.05, 1.25),
        "COMMERCIAL": (0.90, 1.04),
        "INSTANT": (0.812, 0.985),
    }

    def __init__(self, forgetting_factor: float = 0.65, lr: float = 0.45, l2_reg: float = 0.08):
        self.gamma = forgetting_factor
        self.lr = lr
        self.l2_reg = l2_reg
        self.center: Dict[str, float] = {k: v[0] for k, v in self.PRIOR_PARAMS.items()}
        self.slope: Dict[str, float] = {k: v[1] for k, v in self.PRIOR_PARAMS.items()}
        self.sample_weight: Dict[str, float] = {k: 0.0 for k in self.PRIOR_PARAMS}
        self.seg_fill_ratio: Dict[str, float] = {"SPECIALTY": 0.6, "COMMERCIAL": 0.6, "INSTANT": 0.95}
        self.inst_dump_pressure: float = 0.0
        self.records_used: int = 0

    def predict_win_prob(self, segment: str, price_ratio: float) -> float:
        seg = segment if segment in self.center else "COMMERCIAL"
        if seg in ("SPECIALTY", "COMMERCIAL") and self.seg_fill_ratio.get(seg, 0.6) < 0.82:
            if price_ratio <= (1.030 if seg == "COMMERCIAL" else 1.225):
                return 0.92
        c = self.center[seg]
        k = self.slope[seg]
        logit = float(np.clip(-k * (price_ratio - c), -20.0, 20.0))
        return 1.0 / (1.0 + math.exp(-logit))

    def fit_from_memory(self, memory: Any, all_res: List[dict], tv_map: Dict[str, float], rfq_map: Dict[str, dict]):
        rounds_hist = getattr(memory, "rounds", []) or []
        seg_wanted = {"SPECIALTY": 0.0, "COMMERCIAL": 0.0, "INSTANT": 0.0}
        seg_bought = {"SPECIALTY": 0.0, "COMMERCIAL": 0.0, "INSTANT": 0.0}
        avg_market_tv = float(np.mean(list(tv_map.values()))) if tv_map else 110.0

        for rnd in rounds_hist[-2:]:
            if not isinstance(rnd, dict):
                continue
            for s in (rnd.get("market_summary") or []):
                if not isinstance(s, dict):
                    continue
                seg = classify_segment(s)
                w = float(s.get("wanted_kg") or 0.0)
                b = float(s.get("bought_kg") or 0.0)
                n_offers = int(s.get("offers") or 0)
                p_avg = s.get("price_avg")
                p_min = s.get("price_min")
                if seg in seg_wanted and w > 0:
                    seg_wanted[seg] += w
                    seg_bought[seg] += b
                    if p_avg is not None and float(p_avg) > 0 and avg_market_tv > 0:
                        est_market_ratio = float(p_avg) / avg_market_tv
                        if seg == "COMMERCIAL" and 0.95 <= est_market_ratio <= 1.04:
                            self.center["COMMERCIAL"] = 0.7 * self.center["COMMERCIAL"] + 0.3 * est_market_ratio
                        elif seg == "INSTANT":
                            if b >= w * 0.95 and n_offers >= 25 and p_min is not None and float(p_min) > 0:
                                min_ratio = float(p_min) / avg_market_tv
                                target_inst = float(np.clip(min_ratio + 0.015, 0.815, 0.94))
                                self.center["INSTANT"] = 0.5 * self.center["INSTANT"] + 0.5 * target_inst
                                self.inst_dump_pressure = min(1.0, n_offers / 45.0)
                            elif 0.815 <= est_market_ratio <= 0.985:
                                self.center["INSTANT"] = 0.7 * self.center["INSTANT"] + 0.3 * est_market_ratio

        for seg in seg_wanted:
            if seg_wanted[seg] > 0:
                self.seg_fill_ratio[seg] = seg_bought[seg] / seg_wanted[seg]
                if self.seg_fill_ratio[seg] < 0.80:
                    if seg == "COMMERCIAL":
                        self.center["COMMERCIAL"] = max(self.center["COMMERCIAL"], 1.024)
                    elif seg == "SPECIALTY":
                        self.center["SPECIALTY"] = max(self.center["SPECIALTY"], 1.215)

        for res in all_res:
            if not isinstance(res, dict):
                continue
            status = str(res.get("status") or "").upper()
            if any(w in status for w in ("SPEC", "CEILING", "OVER", "REJECT", "INVALID", "OPEN", "PENDING")):
                continue

            lid = str(res.get("lot_id") or "")
            rid = str(res.get("rfq_id") or "")
            tv = float(tv_map.get(lid, 0.0))
            price = float(res.get("price_per_kg") or res.get("price") or res.get("offer_price") or 0.0)
            if tv <= 0 or price <= 0:
                continue

            rfq_obj = rfq_map.get(rid, res)
            seg = classify_segment(rfq_obj)
            if seg not in self.center:
                seg = "COMMERCIAL"

            offered = float(res.get("kg") or res.get("offered_kg") or res.get("qty_kg") or res.get("qty") or 0.0)
            if offered <= 0:
                continue
            filled = float(
                res.get("filled_kg")
                or res.get("sold_kg")
                or res.get("sold_qty")
                or (offered if ("SOLD" in status or "ACCEPTED" in status) and "PARTIAL" not in status else 0.0)
            )
            y = float(np.clip(filled / offered, 0.0, 1.0))
            ratio = price / tv

            for s in self.sample_weight:
                self.sample_weight[s] *= (self.gamma ** 0.25)

            p_pred = self.predict_win_prob(seg, ratio)
            err = y - p_pred
            prior_c, _ = self.PRIOR_PARAMS[seg]
            eff_lr = self.lr / (1.0 + 0.08 * self.sample_weight[seg])
            grad_c = err * self.slope[seg] - self.l2_reg * (self.center[seg] - prior_c)
            low_b, high_b = self.SEG_BOUNDS[seg]
            self.center[seg] = float(np.clip(self.center[seg] + eff_lr * grad_c * 0.015, low_b, high_b))
            self.sample_weight[seg] += 1.0
            self.records_used += 1

    def infer_opponent_shares(self) -> Tuple[float, float]:
        comm_fill = self.seg_fill_ratio.get("COMMERCIAL", 0.6)
        inst_fill = self.seg_fill_ratio.get("INSTANT", 0.95)
        est_dumper = float(np.clip(0.15 + (inst_fill * 0.25) + (self.inst_dump_pressure * 0.30) - ((1.0 - comm_fill) * 0.15), 0.05, 0.85))
        est_sniper = float(np.clip(0.15 + (comm_fill * 0.25), 0.05, 0.65))
        return est_dumper, est_sniper


def warm_start_from_practice(
    clearing_model: OnlineMarketClearingModel, logs_dir: Path, tv_map: Dict[str, float]
) -> int:
    """อ่านผลรอบซ้อม (kind='practice') ของตลาดแข่งจริงที่เพิ่งจบใน 30 นาทีล่าสุด มาตั้งค่าเริ่มต้นให้ R1"""
    if not logs_dir.exists():
        return 0

    now = time.time()
    recent_comp_files = sorted(
        [
            p for p in logs_dir.glob("memory-competition-*.json")
            if (now - p.stat().st_mtime) <= 1800
        ],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not recent_comp_files:
        return 0

    try:
        data = json.loads(recent_comp_files[0].read_text(encoding="utf-8"))
        rounds = data.get("rounds") or []
        practice_rounds = [
            r for r in rounds
            if isinstance(r, dict) and str(r.get("kind", "")).lower() == "practice"
        ]
        if not practice_rounds:
            return 0

        avg_tv = float(np.mean(list(tv_map.values()))) if tv_map else 110.0
        used_cnt = 0
        for rnd in practice_rounds[-2:]:
            for s in (rnd.get("market_summary") or []):
                if not isinstance(s, dict):
                    continue
                seg = classify_segment(s)
                w = float(s.get("wanted_kg") or 0.0)
                b = float(s.get("bought_kg") or 0.0)
                n_offers = int(s.get("offers") or 0)
                p_avg = s.get("price_avg")
                p_min = s.get("price_min")
                if w <= 0 or p_avg is None or float(p_avg) <= 0:
                    continue

                ratio_avg = float(p_avg) / avg_tv
                if seg == "COMMERCIAL" and 0.92 <= ratio_avg <= 1.04:
                    clearing_model.center["COMMERCIAL"] = 0.5 * clearing_model.center["COMMERCIAL"] + 0.5 * ratio_avg
                    used_cnt += 1
                elif seg == "INSTANT":
                    if b >= w * 0.95 and n_offers >= 25 and p_min is not None and float(p_min) > 0:
                        target_inst = float(np.clip((float(p_min) / avg_tv) + 0.015, 0.815, 0.95))
                        clearing_model.center["INSTANT"] = 0.5 * clearing_model.center["INSTANT"] + 0.5 * target_inst
                    elif 0.815 <= ratio_avg <= 0.985:
                        clearing_model.center["INSTANT"] = 0.5 * clearing_model.center["INSTANT"] + 0.5 * ratio_avg
                    used_cnt += 1
        return used_cnt
    except Exception:
        return 0


def evaluate_orders_monte_carlo(
    orders: List[dict],
    my_lots: List[CoffeeLot],
    routes: List[BuyerRoute],
    est_ceilings: Dict[str, float],
    clearing_model: OnlineMarketClearingModel,
    cur_rnd: int,
    tot_rnd: int,
    max_round_cap: int,
    my_rem_bags: int,
    rng: random.Random,
    mc_samples: int = 12,
) -> float:
    if not orders:
        return -9999.0

    route_map = {r.route_id: r for r in routes}
    lot_map = {l.lot_id: l for l in my_lots}
    rounds_left = max(1, tot_rnd - cur_rnd + 1)
    min_rounds_needed = math.ceil(my_rem_bags / max(1, max_round_cap))
    urgency = min(1.2, min_rounds_needed / rounds_left)

    is_boom_upcoming = (tot_rnd >= 15 and cur_rnd < 6)
    if rounds_left <= 1:
        hold_factor = 0.805
    elif urgency >= 0.90:
        hold_factor = 0.85
    elif is_boom_upcoming:
        hold_factor = 1.03
    else:
        hold_factor = 0.94 - (0.04 * min(1.0, cur_rnd / max(1, tot_rnd)))

    total_score = 0.0
    for _ in range(mc_samples):
        sim_ceilings = {
            r.route_id: est_ceilings.get(r.route_id, r.base_ceiling_mult) + rng.uniform(-0.005, 0.004)
            for r in routes
        }

        sim_profit = 0.0
        sold_bags_total = 0.0
        sold_spec_bags = 0.0

        for item in orders:
            rid = item["route_id"]
            lid = item["lot_id"]
            qty = int(item["qty"])
            price = float(item["price"])
            r = route_map.get(rid)
            l = lot_map.get(lid)
            if not r or not l or l.tv <= 0:
                continue

            bid_ratio = price / l.tv
            if bid_ratio > sim_ceilings[rid]:
                continue

            base_prob = clearing_model.predict_win_prob(r.buyer_type, bid_ratio)
            fill_prob = float(np.clip(base_prob + rng.uniform(-0.06, 0.06), 0.05, 1.0))

            expected_fill = qty * fill_prob
            sold_bags_total += expected_fill
            if r.buyer_type == "SPECIALTY":
                sold_spec_bags += expected_fill

            sim_profit += (price - l.tv) * expected_fill

        unsold_bags = max(0.0, my_rem_bags - sold_bags_total)
        avg_tv = float(np.mean([l.tv for l in my_lots if l.bags > 0]))
        continuation_val = unsold_bags * avg_tv * (hold_factor - 1.0)
        spec_bonus = sold_spec_bags * 4.0
        total_score += (sim_profit + continuation_val + spec_bonus)

    return total_score / mc_samples