"""Smart quota-aware fallback planner and batch capacity guard."""
import math
from typing import Any
from history_tracker import classify_segment, extract_history_stats


def bag_weight_kg(rules: dict) -> float:
    for k in ("bag_kg", "kg_per_bag", "bag_weight_kg", "sack_kg", "unit_kg"):
        if k in rules and rules[k] is not None:
            return float(rules[k])
    return 60.0


def get_lot_avail_kg(lot: dict) -> float:
    for k in ("kg_available", "kg_remaining", "available_kg", "remaining_kg"):
        if k in lot and lot[k] is not None:
            return float(lot[k])
    return 0.0


def get_rfq_wanted_kg(rfq: dict, rules: dict) -> float:
    for k in ("wanted_kg", "capacity_left_kg", "remaining_kg", "quota_kg", "kg"):
        if k in rfq and rfq[k] is not None:
            return float(rfq[k])
    bags = float(rfq.get("quota_bags") or rfq.get("bags") or 0)
    return bags * bag_weight_kg(rules)


def get_rfq_min_kg(rfq: dict, rules: dict) -> float:
    for k in ("min_kg", "min_order_kg", "min_lot_kg"):
        if k in rfq and rfq[k] is not None:
            return float(rfq[k])
    bags = float(rfq.get("min_bags") or 0)
    return bags * bag_weight_kg(rules) if bags > 0 else bag_weight_kg(rules)


def candidates(state: dict, ctx: Any) -> list[dict]:
    bag_kg = bag_weight_kg(ctx.rules)
    rfqs = state.get("rfqs") or []
    rfq_map = {r["rfq_id"]: r for r in rfqs if r.get("rfq_id")}
    _, _, _, banned_segments = extract_history_stats(state, ctx.memory, ctx.tv, rfq_map)

    rnd_obj = state.get("round") or {}
    cur_rnd = int(rnd_obj.get("round_num") or rnd_obj.get("number") or 1)
    tot_rnd = int(rnd_obj.get("total_rounds") or ctx.rules.get("total_rounds") or 9)
    is_endgame = (tot_rnd - cur_rnd + 1) <= 2

    pairs = []
    for l in (state.get("lots") or []):
        lid = l.get("lot_id")
        avail = get_lot_avail_kg(l)
        if not lid or avail < bag_kg:
            continue
        s = l.get("sample") or l
        d = float(s.get("defect_pct") or 5.0)
        m = float(s.get("moisture_pct") or 12.0)
        sc = float(s.get("screen") or 15.0)

        if d <= 7.0:
            d += (0.25 if is_endgame else 0.65)
        if m <= 13.5 and not is_endgame:
            m += 0.25

        banned = banned_segments.get(lid, set())
        if "COMMERCIAL" in banned:
            d = max(d, 8.5)
        elif "SPECIALTY" in banned:
            d = max(d, 3.8)

        for r in rfqs:
            rid = r.get("rfq_id")
            wanted = get_rfq_wanted_kg(r, ctx.rules)
            min_req = get_rfq_min_kg(r, ctx.rules)
            if not rid or wanted < min_req or avail < min_req:
                continue

            seg = classify_segment(r)
            spec = r.get("spec") or r.get("requirements") or r
            def_d = 3.0 if seg == "SPECIALTY" else (7.0 if seg == "COMMERCIAL" else 15.0)
            def_m = 12.5 if seg in ("SPECIALTY", "COMMERCIAL") else 15.0
            buyer = str(r.get("buyer") or r.get("buyer_name") or rid)
            def_s = 17.0 if "บางกอกบลูม" in buyer else (16.0 if seg == "SPECIALTY" else 0.0)

            max_d = float(spec.get("max_defect_pct") or spec.get("max_defect") or def_d)
            max_m = float(spec.get("max_moisture_pct") or spec.get("max_moisture") or def_m)
            min_s = float(spec.get("min_screen") or spec.get("min_size") or def_s)

            if d <= max_d and m <= max_m and sc >= min_s:
                pairs.append({"lot": l, "rfq": r, "segment": seg, "min_screen": min_s})

    # เรียงลำดับความสำคัญ: Specialty ก่อนเสมอ + เอาล็อต TV สูงสุดเข้าโควตารอบก่อน
    pairs.sort(
        key=lambda p: (
            0 if p["segment"] == "SPECIALTY" else (1 if p["segment"] == "COMMERCIAL" else 2),
            -float(ctx.tv.get(p["lot"]["lot_id"], 100.0)),
        )
    )
    return pairs


def rule_plan(state: dict, ctx: Any, cand_list: list[dict]) -> list[dict]:
    bag_kg = bag_weight_kg(ctx.rules)
    rfqs = state.get("rfqs") or []
    rfq_map = {r["rfq_id"]: r for r in rfqs if r.get("rfq_id")}
    _, _, est_ceilings, _ = extract_history_stats(state, ctx.memory, ctx.tv, rfq_map)

    rnd_obj = state.get("round") or {}
    cur_rnd = int(rnd_obj.get("round_num") or rnd_obj.get("number") or 1)
    tot_rnd = int(rnd_obj.get("total_rounds") or ctx.rules.get("total_rounds") or 9)
    rounds_left = max(1, tot_rnd - cur_rnd + 1)
    urgency = min(1.0, cur_rnd / max(1, tot_rnd))

    cap_left = float(state.get("capacity_left_kg") or state.get("round_capacity_kg") or 300.0)
    lot_used: dict[str, float] = {}
    rfq_used: dict[str, float] = {}
    offers: list[dict] = []

    for pair in cand_list:
        if cap_left < bag_kg:
            break
        l, r = pair["lot"], pair["rfq"]
        lid, rid = l["lot_id"], r["rfq_id"]
        min_req = get_rfq_min_kg(r, ctx.rules)
        avail = get_lot_avail_kg(l) - lot_used.get(lid, 0.0)
        wanted = get_rfq_wanted_kg(r, ctx.rules) - rfq_used.get(rid, 0.0)
        take = min(avail, wanted, cap_left)
        take = (take // bag_kg) * bag_kg
        if take >= min_req and take >= bag_kg:
            seg = pair.get("segment") or classify_segment(r)
            ceil = est_ceilings.get(rid, 1.22 if seg == "SPECIALTY" else (1.028 if seg == "COMMERCIAL" else 0.978))

            if seg == "SPECIALTY":
                base_mult = 1.185 if pair.get("min_screen", 0) >= 17 else 1.125
                mult = min(ceil - 0.004, max(1.05, base_mult - 0.03 * urgency))
            elif seg == "COMMERCIAL":
                base_mult = 1.010 - (0.045 * urgency) - (0.02 if rounds_left <= 2 else 0.0)
                mult = min(ceil - 0.004, max(0.92, base_mult))
            else:
                base_mult = 0.955 - (0.055 * urgency) - (0.03 if rounds_left <= 2 else 0.0)
                mult = min(ceil - 0.004, max(0.84, base_mult))

            offers.append({
                "rfq_id": rid,
                "lot_id": lid,
                "kg": take,
                "price_per_kg": round(ctx.tv.get(lid, 100.0) * mult, 2),
            })
            lot_used[lid] = lot_used.get(lid, 0.0) + take
            rfq_used[rid] = rfq_used.get(rid, 0.0) + take
            cap_left -= take
    return offers


def guard(state: dict, ctx: Any, offers: list[dict]) -> list[dict]:
    rules = ctx.rules
    lots = {l["lot_id"]: l for l in (state.get("lots") or []) if l.get("lot_id")}
    rfqs = {r["rfq_id"]: r for r in (state.get("rfqs") or []) if r.get("rfq_id")}

    lot_rem = {lid: get_lot_avail_kg(l) for lid, l in lots.items()}
    rfq_rem = {rid: get_rfq_wanted_kg(r, rules) for rid, r in rfqs.items()}
    round_cap_rem = float(state.get("capacity_left_kg") or state.get("round_capacity_kg") or 999999.0)

    clean: list[dict] = []
    for o in offers:
        if round_cap_rem <= 0:
            break
        rid = o.get("rfq_id")
        lid = o.get("lot_id")
        if rid not in rfqs or lid not in lots:
            continue

        min_kg = get_rfq_min_kg(rfqs[rid], rules)
        kg = min(float(o.get("kg", 0.0)), lot_rem[lid], rfq_rem[rid], round_cap_rem)
        if kg < min_kg or kg <= 0:
            continue

        lot_tv = float(ctx.tv.get(lid, 100.0))
        floor_price = round(lot_tv * 0.795, 2)
        raw_p = float(o.get("price_per_kg") or o.get("price") or lot_tv)
        price_per_kg = max(floor_price, round(raw_p, 2))

        clean.append({
            "rfq_id": rid,
            "lot_id": lid,
            "kg": round(kg, 2),
            "price_per_kg": price_per_kg,
        })
        lot_rem[lid] = max(0.0, lot_rem[lid] - kg)
        rfq_rem[rid] = max(0.0, rfq_rem[rid] - kg)
        round_cap_rem = max(0.0, round_cap_rem - kg)

    return clean
