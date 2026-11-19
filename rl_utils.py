"""Utilities for loading PPO models, parsing server state, and building 21-dim observations."""
import io
import math
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from stable_baselines3 import PPO

from coffee_math import CoffeeLot, BuyerRoute
from action_mask import is_lot_eligible_for_route
from history_tracker import classify_segment
from fallback_rules import bag_weight_kg, get_lot_avail_kg, get_rfq_wanted_kg, get_rfq_min_kg

HERE = Path(__file__).resolve().parent


def load_ppo_model(folder_name: str, zip_name: str):
    p_dir = HERE / "models" / folder_name
    p_zip = HERE / "models" / zip_name
    if p_dir.exists() and p_dir.is_dir():
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for f in p_dir.iterdir():
                if f.is_file():
                    zf.write(f, f.name)
        buf.seek(0)
        print(f"✅ โหลดโมเดล RL สำเร็จ (จากโฟลเดอร์): {p_dir}", flush=True)
        return PPO.load(buf, device="cpu")
    if p_zip.exists():
        print(f"✅ โหลดโมเดล RL สำเร็จ (จากไฟล์): {p_zip}", flush=True)
        return PPO.load(str(p_zip), device="cpu")
    print(f"⚠️ ไม่พบโมเดล: {p_dir} หรือ {p_zip}", flush=True)
    return None


def get_round_progress(state: dict, ctx: Any) -> Tuple[int, int]:
    rnd_obj = state.get("round") or {}
    cur_rnd = int(rnd_obj.get("number") or rnd_obj.get("round_num") or rnd_obj.get("index") or 0)
    rid_str = str(rnd_obj.get("round_id") or "")
    if cur_rnd <= 0 and rid_str:
        digits = "".join(ch for ch in rid_str if ch.isdigit())
        cur_rnd = int(digits) if digits else 1
    cur_rnd = max(1, cur_rnd)

    # เซิร์ฟเวอร์ส่งจำนวนรอบรวมมาใน key "of" (เช่น {"number": 2, "of": 15})
    rules_tot = int(
        rnd_obj.get("of")
        or rnd_obj.get("total_rounds")
        or ctx.rules.get("total_rounds")
        or ctx.rules.get("real_rounds")
        or 0
    )

    if rules_tot <= 0 and isinstance(ctx.calendar, list):
        real_rounds = [c for c in ctx.calendar if isinstance(c, dict) and str(c.get("kind", "real")).lower() != "practice"]
        rules_tot = len(real_rounds) if real_rounds else len(ctx.calendar)

    if rules_tot > 0:
        tot_rnd = max(cur_rnd, rules_tot)
    else:
        tot_rnd = 15 if str(state.get("market", "")).lower() == "competition" or cur_rnd > 8 else 8

    return cur_rnd, tot_rnd


def estimate_event_phase(cur_rnd: int, tot_rnd: int, ctx: Any, state: dict = None, routes: List[BuyerRoute] = None) -> Tuple[float, int]:
    # 1. อ่านป้ายกำกับปฏิทินโดยตรงจาก state["round"]["calendar"]
    rnd_obj = (state or {}).get("round") or {}
    cal_label = str(rnd_obj.get("calendar") or "").lower()
    if any(w in cal_label for w in ("เทศกาล", "festival", "specialty", "peak")):
        return 1.15, 1
    if any(w in cal_label for w in ("สุดท้าย", "final", "ต่ำสุด")):
        return 0.28, 3
    if any(w in cal_label for w in ("ใกล้ปิด", "ลดลง", "low")):
        return 0.55, 2

    # 2. อ่านจาก ctx.calendar
    if isinstance(ctx.calendar, list) and len(ctx.calendar) >= cur_rnd:
        entry = ctx.calendar[cur_rnd - 1]
        if isinstance(entry, dict):
            ev = str(entry.get("calendar") or entry.get("event") or entry.get("theme") or "").lower()
            if any(w in ev for w in ("เทศกาล", "festival", "specialty", "peak")):
                return 1.15, 1
            if any(w in ev for w in ("สุดท้าย", "final", "ต่ำสุด")):
                return 0.28, 3
            if any(w in ev for w in ("ใกล้ปิด", "ลดลง", "low")):
                return 0.55, 2

    # 3. ตรวจจับจากหน้างานจริง
    if routes:
        spec_q = sum(r.quota for r in routes if r.buyer_type == "SPECIALTY")
        tot_q = sum(r.quota for r in routes)
        if tot_q > 0 and (spec_q / tot_q) >= 0.28:
            return 1.15, 1

    # 4. ตารางมาตรฐาน 15 รอบจริง
    if tot_rnd >= 15:
        if 6 <= cur_rnd <= 8:
            return 1.15, 1
        if 13 <= cur_rnd <= 14:
            return 0.55, 2
        if cur_rnd >= 15:
            return 0.28, 3
    return 1.00, 0


def parse_lots_and_routes(
    state: dict,
    ctx: Any,
    banned_segments: Dict[str, set],
    is_endgame: bool,
    defect_buf_normal: float = 0.20,
    defect_buf_endgame: float = 0.0,
    moisture_buf: float = 0.0,
) -> Tuple[List[CoffeeLot], List[BuyerRoute], int, int, float]:
    bag_kg = bag_weight_kg(ctx.rules)
    raw_lots = state.get("lots") or []
    raw_rfqs = state.get("rfqs") or []

    my_lots: List[CoffeeLot] = []
    my_init_bags = 0
    for l in raw_lots:
        lid = l.get("lot_id")
        if not lid:
            continue
        avail_bags = int(get_lot_avail_kg(l) // bag_kg)
        init_kg = next((float(l[k]) for k in ("kg_total", "initial_kg", "total_kg", "kg") if l.get(k) is not None), get_lot_avail_kg(l))
        init_bags = max(avail_bags, int(init_kg // bag_kg))
        my_init_bags += init_bags

        sample = l.get("sample") or l
        tv_val = float(ctx.tv.get(lid, 100.0))
        raw_s = float(sample.get("screen") or sample.get("screen_size") or 15.0)
        raw_m = float(sample.get("moisture_pct") or sample.get("moisture") or 12.0)
        raw_g = float(sample.get("defect_pct") or sample.get("defect") or 5.0)

        obs_m = (raw_m + moisture_buf) if (raw_m <= 13.5 and not is_endgame) else raw_m
        obs_g = (raw_g + (defect_buf_endgame if is_endgame else defect_buf_normal)) if raw_g <= 7.0 else raw_g

        banned = banned_segments.get(lid, set())
        if "COMMERCIAL" in banned:
            obs_g = max(obs_g, 8.5)
        elif "SPECIALTY" in banned:
            obs_g = max(obs_g, 3.8)

        my_lots.append(CoffeeLot(lot_id=lid, bags=avail_bags, s=raw_s, m=obs_m, g=obs_g, tv=tv_val))

    routes: List[BuyerRoute] = []
    for r in raw_rfqs:
        rid = r.get("rfq_id")
        if not rid:
            continue
        seg = classify_segment(r)
        spec = r.get("spec") or r.get("requirements") or r
        def_d = 3.0 if seg == "SPECIALTY" else (7.0 if seg == "COMMERCIAL" else 15.0)
        def_m = 12.5 if seg in ("SPECIALTY", "COMMERCIAL") else 15.0
        buyer = str(r.get("buyer") or r.get("buyer_name") or rid)
        def_s = 17.0 if "บางกอกบลูม" in buyer else (16.0 if seg == "SPECIALTY" else 0.0)

        routes.append(BuyerRoute(
            route_id=rid,
            company_name=buyer,
            buyer_type=seg,
            quota=int(get_rfq_wanted_kg(r, ctx.rules) // bag_kg),
            max_defect=float(spec.get("max_defect_pct") or spec.get("max_defect") or def_d),
            max_moisture=float(spec.get("max_moisture_pct") or spec.get("max_moisture") or def_m),
            min_screen=float(spec.get("min_screen") or spec.get("min_size") or def_s),
            min_bags_per_bid=max(1, int(math.ceil(get_rfq_min_kg(r, ctx.rules) / bag_kg))),
        ))

    my_rem_bags = sum(l.bags for l in my_lots)
    round_cap_kg = float(state.get("capacity_left_kg") or state.get("round_capacity_kg") or (my_rem_bags * bag_kg))
    max_round_cap = max(1, int(round_cap_kg // bag_kg))

    return my_lots, routes, my_init_bags, max_round_cap, bag_kg


def build_obs_21(
    state: dict,
    ctx: Any,
    my_lots: List[CoffeeLot],
    routes: List[BuyerRoute],
    my_init_bags: int,
    max_round_cap: int,
    cur_rnd: int,
    tot_rnd: int,
    last_win: float,
    last_exc: float,
    est_dumper_share: float,
    est_sniper_share: float,
) -> Tuple[np.ndarray, float]:
    my_rem_bags = sum(l.bags for l in my_lots)
    total_demand = sum(r.quota for r in routes)
    num_teams_est = max(1, int(state.get("num_teams") or len(state.get("standings") or []) or 55))

    if total_demand > my_rem_bags * 4:
        all_init_bags_est = max(my_init_bags, my_init_bags * num_teams_est)
        all_rem_bags_est = max(my_rem_bags, my_rem_bags * num_teams_est)
    else:
        all_init_bags_est = max(1, my_init_bags)
        all_rem_bags_est = max(1, my_rem_bags)

    spec_eligible_quota = sum(
        r.quota for r in routes
        if r.buyer_type == "SPECIALTY" and any(is_lot_eligible_for_route(l, r, 0.0) for l in my_lots if l.bags > 0)
    )
    avg_tv_norm = float(np.mean([l.tv for l in my_lots if l.bags > 0]) / 135.0)
    spec_bags = sum(l.bags for l in my_lots if l.bags > 0 and l.s >= 16.0 and l.g <= 3.0 and l.m <= 12.5)

    demand_mult, phase_code = estimate_event_phase(cur_rnd, tot_rnd, ctx, state, routes)
    # 15 รอบจริงทุกตัวเป็นรอบประมูล (ไม่มี R16 แยก)
    auction_rounds = tot_rnd
    auction_rounds_left = max(1, auction_rounds - cur_rnd + 1)
    min_rounds_needed = math.ceil(my_rem_bags / max(1, max_round_cap))
    cap_urgency = min(1.5, min_rounds_needed / max(1, auction_rounds_left))

    spec_spike_readiness = min(1.5, spec_bags / max(1, 3 * max_round_cap))
    supply_depletion = float(np.clip(1.0 - (all_rem_bags_est / max(1, all_init_bags_est)), 0.0, 1.0))

    # Map 15 รอบจริงให้ตรงกับ scale 16 รอบที่โมเดลเทรนมา
    env_tot = 16.0 if tot_rnd >= 15 else float(tot_rnd)
    obs_21 = np.array([
        cur_rnd / env_tot,
        auction_rounds_left / max(1.0, float(auction_rounds)),
        my_rem_bags / max(1, my_init_bags),
        min(1.0, my_rem_bags / max(1, max_round_cap)),
        min(2.0, total_demand / max(1, all_rem_bags_est)),
        avg_tv_norm,
        spec_bags / max(1, my_rem_bags),
        min(1.0, spec_eligible_quota / max(1, max_round_cap)),
        sum(r.quota for r in routes if r.buyer_type == "COMMERCIAL") / max(1, total_demand),
        sum(r.quota for r in routes if r.buyer_type == "INSTANT") / max(1, total_demand),
        1.0 if tot_rnd >= 15 else 0.0,
        phase_code / 4.0,
        min(1.0, spec_eligible_quota / max(1, total_demand)),
        last_win,
        last_exc,
        demand_mult / 1.5,
        cap_urgency,
        spec_spike_readiness,
        supply_depletion,
        est_dumper_share,
        est_sniper_share,
    ], dtype=np.float32)

    return obs_21, cap_urgency