"""Track buyer price ceilings, outbid signals, and spec-failed lots from past rounds."""
from typing import Any


def classify_segment(rfq: dict) -> str:
    raw = str(rfq.get("buyer_type") or rfq.get("segment") or rfq.get("category") or "").upper()
    name = str(rfq.get("buyer") or rfq.get("buyer_name") or rfq.get("company_name") or "")
    if "SPEC" in raw or "บางกอกบลูม" in name or "ดอยช้าง" in name or "เชียงใหม่" in name:
        return "SPECIALTY"
    if "COMM" in raw or any(k in name for k in ("พรีเมียม", "โกลบอล", "สยาม", "ชุมพร", "ไทยโรสเตอร์", "อาราบิก้า", "บูรพา")):
        return "COMMERCIAL"
    if "INST" in raw:
        return "INSTANT"
    spec = rfq.get("spec") or rfq.get("requirements") or rfq
    max_d = float(spec.get("max_defect_pct") or spec.get("max_defect") or 15.0)
    if max_d <= 3.5:
        return "SPECIALTY"
    if max_d <= 7.5:
        return "COMMERCIAL"
    return "INSTANT"


def _collect_all_result_records(state: dict, memory: Any) -> list[dict]:
    """รวบรวมผลลัพธ์จากทุก Key ที่เป็นไปได้ทั้งใน state และ memory.rounds"""
    all_res: list[dict] = []
    rounds_hist = getattr(memory, "rounds", []) or []
    for rnd in rounds_hist:
        if not isinstance(rnd, dict):
            continue
        for key in ("results", "my_results", "my_offers", "offers", "outcomes"):
            items = rnd.get(key)
            if isinstance(items, list):
                all_res.extend(x for x in items if isinstance(x, dict))

    for key in ("last_results", "results", "my_results", "round_results"):
        items = state.get(key)
        if isinstance(items, list):
            all_res.extend(x for x in items if isinstance(x, dict))

    return all_res


def extract_history_stats(
    state: dict, memory: Any, tv_map: dict[str, float], rfq_map: dict[str, dict]
) -> tuple[float, float, dict[str, float], dict[str, set[str]]]:
    type_defaults = {"SPECIALTY": 1.225, "COMMERCIAL": 1.028, "INSTANT": 0.978}
    ceilings: dict[str, float] = {}
    company_ceilings: dict[str, float] = {}

    for rid, r in rfq_map.items():
        seg = classify_segment(r)
        spec = r.get("spec") or r.get("requirements") or r
        min_s = float(spec.get("min_screen") or spec.get("min_size") or 0.0)
        ceilings[rid] = 1.235 if (seg == "SPECIALTY" and min_s >= 17) else type_defaults.get(seg, 1.00)

    banned_segments: dict[str, set[str]] = {}
    sent_map: dict[tuple[str, str], dict] = {}
    if hasattr(memory, "sent") and isinstance(memory.sent, dict):
        for _, offers_list in memory.sent.items():
            for s_item in offers_list:
                if isinstance(s_item, dict):
                    sent_map[(str(s_item.get("rfq_id")), str(s_item.get("lot_id")))] = s_item

    all_res = _collect_all_result_records(state, memory)

    for res in all_res:
        lid = str(res.get("lot_id") or "")
        rid = str(res.get("rfq_id") or "")
        buyer = str(res.get("buyer") or res.get("buyer_name") or res.get("company_name") or "")
        full_status = " ".join(
            str(res.get(k) or "") for k in ("status", "reason", "reject_reason", "error", "message", "note")
        ).upper()
        price = float(res.get("price_per_kg") or res.get("price") or res.get("offer_price") or 0.0)
        lot_tv = float(tv_map.get(lid, 0.0))

        # 1. ตรวจจับราคาเกินเพดานผู้ซื้อ
        is_ceiling = any(w in full_status for w in ("CEILING", "CAP", "HIGH", "เกินเพดาน", "MAX_PRICE", "OVER_CEILING"))
        if is_ceiling and lot_tv > 0 and price > 0:
            ratio = price / lot_tv
            new_cap = round(ratio - 0.003, 4)
            if rid in ceilings:
                ceilings[rid] = min(ceilings[rid], new_cap)
            if buyer:
                company_ceilings[buyer] = min(company_ceilings.get(buyer, 1.35), new_cap)

        # 2. ตรวจจับล็อตที่ไม่ผ่านเกณฑ์คุณภาพ (Spec Failed)
        is_outbid = any(w in full_status for w in ("OUTBID", "QUOTA", "FULL", "SOLD", "ACCEPTED", "OPEN", "PENDING", "PARTIAL"))
        is_spec_fail = any(w in full_status for w in ("SPEC", "DEFECT", "MOISTURE", "SCREEN", "QUALITY", "FAIL", "REJECT", "INVALID", "ไม่ผ่าน"))
        if lid and is_spec_fail and not is_outbid and not is_ceiling:
            rfq_obj = rfq_map.get(rid)
            sent_info = sent_map.get((rid, lid), {})
            seg = classify_segment(rfq_obj) if rfq_obj else str(sent_info.get("buyer_type") or res.get("buyer_type") or "").upper()
            if "SPEC" in seg or "บางกอก" in buyer or "ดอยช้าง" in buyer:
                banned_segments.setdefault(lid, set()).add("SPECIALTY")
            elif "COMM" in seg:
                banned_segments.setdefault(lid, set()).update({"SPECIALTY", "COMMERCIAL"})

    # ซิงก์เพดานระดับชื่อบริษัทเข้ากับ RFQ ของรอบปัจจุบันด้วย
    for rid, r in rfq_map.items():
        b_name = str(r.get("buyer") or r.get("buyer_name") or r.get("company_name") or "")
        if b_name in company_ceilings:
            ceilings[rid] = min(ceilings[rid], company_ceilings[b_name])

    # 3. คำนวณ Win Rate และ Ceiling Exceed Rate ของรอบล่าสุด
    last_win, last_exc = 0.5, 0.0
    rounds_hist = getattr(memory, "rounds", []) or []
    recent_records = []
    if isinstance(state.get("last_results"), list) and state["last_results"]:
        recent_records = state["last_results"]
    elif rounds_hist and isinstance(rounds_hist[-1], dict):
        recent_records = (
            rounds_hist[-1].get("results")
            or rounds_hist[-1].get("my_results")
            or rounds_hist[-1].get("my_offers")
            or []
        )

    if recent_records:
        sub_kg = 0.0
        sold_kg = 0.0
        exc_kg = 0.0
        for o in recent_records:
            if not isinstance(o, dict):
                continue
            offered = float(o.get("kg") or o.get("offered_kg") or o.get("qty_kg") or 0.0)
            st = str(o.get("status") or "").upper()
            filled = float(
                o.get("filled_kg")
                or o.get("sold_kg")
                or (offered if ("SOLD" in st and "PARTIAL" not in st) else 0.0)
            )
            sub_kg += offered
            sold_kg += filled
            if any(w in st for w in ("CEILING", "HIGH", "เกินเพดาน")):
                exc_kg += offered

        if sub_kg > 0:
            last_win = min(1.0, sold_kg / sub_kg)
            last_exc = min(1.0, exc_kg / sub_kg)

    return last_win, last_exc, ceilings, banned_segments
