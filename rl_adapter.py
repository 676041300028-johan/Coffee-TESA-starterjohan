"""Main RL Strategy Controller: Market-Benchmark Pricing & Progressive Endgame Price Decay."""
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np

from bag_router import BagCombinatorialRouter
from history_tracker import extract_history_stats, _collect_all_result_records
from rl_utils import (
    load_ppo_model,
    get_round_progress,
    parse_lots_and_routes,
    build_obs_21,
)
from market_learner import (
    OnlineMarketClearingModel,
    evaluate_orders_monte_carlo,
    warm_start_from_practice,
)

HERE = Path(__file__).resolve().parent

DEFECT_BUFFER_NORMAL = 0.20
DEFECT_BUFFER_ENDGAME = 0.0
MOISTURE_BUFFER = 0.0
ROLLOUT_NUM_CANDIDATES = 21
ROLLOUT_MC_SAMPLES = 12

RL_MODEL_16 = load_ppo_model("rl_16", "coffee_rl_16rounds_event_final.zip")
RL_MODEL_9 = load_ppo_model("rl_9", "coffee_rl_9rounds_final.zip")
ROUTER = BagCombinatorialRouter(error_buffer=0.0)


def generate_candidate_actions(
    base_sell_ratio: float,
    base_comm_mult: float,
    base_inst_mult: float,
    is_endgame: bool,
    cap_urgency: float,
    last_win: float,
    last_exc: float,
) -> List[Tuple[str, float, float, float]]:
    if is_endgame or cap_urgency >= 0.65:
        sell_grid = [1.0]
    else:
        sell_grid = [
            float(np.clip(base_sell_ratio + 0.15, 0.85, 1.0)),
            1.0,
        ]

    up_shift = 0.002 if last_exc > 0.15 else (0.008 if last_win >= 0.85 else 0.003)
    price_shifts = [
        ("rl_base", 0.000, 0.000),
        ("agg_profit", up_shift, up_shift * 0.8),
        ("slight_cut", -0.014, -0.025),
        ("med_cut", -0.028, -0.045),
        ("deep_cut", -0.045, -0.070),
        ("panic_dump", -0.060, -0.095),
        ("split_guard", 0.003, -0.055),
    ]

    candidates: List[Tuple[str, float, float, float]] = []
    seen = set()
    for sr in sell_grid:
        for label, d_comm, d_inst in price_shifts:
            c_m = round(float(np.clip(base_comm_mult + d_comm, 0.84, 1.015)), 4)
            i_m = round(float(np.clip(base_inst_mult + d_inst, 0.810, 0.975)), 4)
            key = (round(sr, 3), c_m, i_m)
            if key not in seen:
                seen.add(key)
                candidates.append((f"{label}@sr={sr:.2f}", sr, c_m, i_m))
            if len(candidates) >= ROLLOUT_NUM_CANDIDATES:
                return candidates
    return candidates


def run_rl_agent(state: dict, ctx: Any) -> Tuple[List[dict], str]:
    cur_rnd, tot_rnd = get_round_progress(state, ctx)
    model = RL_MODEL_9 if (tot_rnd <= 11 and RL_MODEL_9 is not None) else (RL_MODEL_16 or RL_MODEL_9)
    tag = "ppo_rl_9r" if model is RL_MODEL_9 else "ppo_rl_16r"
    if model is None:
        return [], "no_model"

    raw_rfqs = state.get("rfqs") or []
    rfq_map = {r["rfq_id"]: r for r in raw_rfqs if r.get("rfq_id")}
    last_win, last_exc, est_ceilings, banned_segments = extract_history_stats(state, ctx.memory, ctx.tv, rfq_map)

    all_records = _collect_all_result_records(state, ctx.memory)
    clearing_model = OnlineMarketClearingModel()
    if str(state.get("market", "")).lower() == "competition" and not all_records:
        warm_start_from_practice(clearing_model, HERE / "logs", ctx.tv)
    clearing_model.fit_from_memory(ctx.memory, all_records, ctx.tv, rfq_map)
    est_dumper, est_sniper = clearing_model.infer_opponent_shares()

    is_endgame = ((tot_rnd - cur_rnd + 1) <= 2)
    my_lots, routes, my_init_bags, max_round_cap, bag_kg = parse_lots_and_routes(
        state, ctx, banned_segments, is_endgame, DEFECT_BUFFER_NORMAL, DEFECT_BUFFER_ENDGAME, MOISTURE_BUFFER
    )

    my_rem_bags = sum(l.bags for l in my_lots)
    if my_rem_bags <= 0 or not routes:
        return [], tag

    obs_21, cap_urgency = build_obs_21(
        state, ctx, my_lots, routes, my_init_bags, max_round_cap,
        cur_rnd, tot_rnd, last_win, last_exc, est_dumper, est_sniper
    )
    obs = obs_21[:model.observation_space.shape[0]]

    action, _ = model.predict(obs, deterministic=True)
    base_sr = float(np.clip(action[0], 0.0, 1.0))
    comm_sig = float(np.clip(action[1], 0.0, 1.0))
    inst_sig = float(np.clip(action[2], 0.0, 1.0)) if len(action) > 2 else comm_sig

    # Progress Decay: ยิ่งเข้าใกล้รอบท้ายๆ ยิ่งลดราคาลงเรื่อยๆ (สูงสุด -0.055x TV ในรอบสุดท้าย)
    progress = min(1.0, max(0.0, (cur_rnd - 1) / max(1, tot_rnd - 1)))
    round_decay = 0.055 * progress

    mkt_comm = clearing_model.center.get("COMMERCIAL", 0.955)
    mkt_inst = clearing_model.center.get("INSTANT", 0.895)

    # ล็อกฐานราคาให้ไม่สูงกว่าค่าเฉลี่ยตลาดจริง และลดลงตาม progress ของรอบ
    base_comm = min(mkt_comm + 0.003, 0.955 + (comm_sig * 0.035) - round_decay)
    base_inst = (0.825 + (inst_sig * 0.04)) if is_endgame else min(mkt_inst + 0.003, 0.905 + (inst_sig * 0.045) - round_decay)

    candidates = generate_candidate_actions(base_sr, base_comm, base_inst, is_endgame, cap_urgency, last_win, last_exc)

    sim_rng = ctx.rng if getattr(ctx, "rng", None) is not None else random.Random(cur_rnd * 997)
    best_routed: List[dict] = []
    best_score = -1e18
    best_label = "rl_base"

    for label, sr, c_mult, i_mult in candidates:
        t_bags = max(min(my_rem_bags, max_round_cap), int(round(sr * max_round_cap))) if (is_endgame or cap_urgency >= 0.65) else int(round(sr * max_round_cap))
        cand_routed = ROUTER.route_bags(
            my_lots, routes, t_bags, c_mult, est_ceilings,
            inst_price_multiplier=i_mult, protect_spec=False, max_spike_reserve_bags=0
        )
        if not cand_routed:
            continue
        score = evaluate_orders_monte_carlo(
            cand_routed, my_lots, routes, est_ceilings, clearing_model,
            cur_rnd, tot_rnd, max_round_cap, my_rem_bags, sim_rng, ROLLOUT_MC_SAMPLES
        )
        if score > best_score:
            best_score, best_routed, best_label = score, cand_routed, label

    routed = best_routed

    # เติมให้เต็มเพดาน max_round_cap เสมอ
    used_lot: Dict[str, int] = {}
    used_route: Dict[str, int] = {}
    for item in routed:
        q = int(item.get("qty") or 0)
        used_lot[item["lot_id"]] = used_lot.get(item["lot_id"], 0) + q
        used_route[item["route_id"]] = used_route.get(item["route_id"], 0) + q

    rem_cap = max_round_cap - sum(used_lot.values())
    if rem_cap > 0:
        for l in sorted(my_lots, key=lambda x: x.tv, reverse=True):
            rem_l = l.bags - used_lot.get(l.lot_id, 0)
            for r in sorted(routes, key=lambda x: x.quota, reverse=True):
                rem_q = r.quota - used_route.get(r.route_id, 0)
                min_req = max(1, r.min_bags_per_bid)
                if rem_l >= min_req and rem_q >= min_req and rem_cap >= min_req:
                    if l.g <= r.max_defect and l.m <= r.max_moisture and l.s >= r.min_screen:
                        take = min(rem_l, rem_q, rem_cap)
                        fill_mult = 0.812 if is_endgame else (min(mkt_comm - 0.005, 0.935) if r.buyer_type == "COMMERCIAL" else min(mkt_inst - 0.005, 0.875))
                        dump_mult = min(est_ceilings.get(r.route_id, 0.978) - 0.003, fill_mult)
                        routed.append({"lot_id": l.lot_id, "route_id": r.route_id, "qty": take, "price": round(l.tv * dump_mult, 2)})
                        rem_l -= take
                        rem_cap -= take
                        used_route[r.route_id] = used_route.get(r.route_id, 0) + take

    offers: List[dict] = []
    tot_val, tot_tv = 0.0, 0.0
    tv_lookup = {l.lot_id: l.tv for l in my_lots}
    for item in routed:
        qty_bags = int(item.get("qty") or 0)
        if qty_bags > 0:
            kg_v, p_v = round(qty_bags * bag_kg, 2), round(float(item["price"]), 2)
            offers.append({"rfq_id": item["route_id"], "lot_id": item["lot_id"], "kg": kg_v, "price_per_kg": p_v})
            tot_val += kg_v * p_v
            tot_tv += kg_v * tv_lookup.get(item["lot_id"], p_v)

    avg_pct = (tot_val / tot_tv * 100.0) if tot_tv > 0 else 0.0
    note_msg = f"R{cur_rnd}/{tot_rnd} | {best_label} | EV={best_score:+.1f} | avg={avg_pct:.1f}%TV | comm_c={mkt_comm:.3f} | win={last_win:.2f}"
    ctx.notes.append(note_msg)
    print(f"   🤖 {note_msg}", flush=True)

    return offers, f"{tag}_rollout"