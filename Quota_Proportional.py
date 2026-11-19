import math
from typing import Dict, List, Any


def calc_tv_and_safe_specs(s: float, m: float, g: float, p0: float = 120.0) -> Dict[str, Any]:
    """
    คำนวณ TV จากค่าที่เห็นบนจอ และคำนวณค่า Worst-Case เผื่อ Error 0.5%
    เพื่อป้องกันส่งของไปแล้วโดนตีตก 'ไม่ผ่านสเปก'
    """
    if g <= 1.0:
        g_mult = 1.08
    elif g <= 3.0:
        g_mult = 1.04
    elif g <= 5.0:
        g_mult = 1.00
    elif g <= 7.0:
        g_mult = 0.96
    elif g <= 15.0:
        g_mult = 0.85
    else:
        g_mult = 0.60

    G = round((1.0 - g / 100.0) * g_mult, 4)

    if m <= 12.5:
        M = 1.0000
    elif m <= 15.0:
        M = round((1.0 - m / 100.0) / 0.875 - 0.02 * (m - 12.5), 4)
    else:
        M = 0.6000

    s_int = int(round(s))
    if s_int <= 12:
        S = 0.92
    elif s_int == 13:
        S = 0.97
    elif s_int in (14, 15):
        S = 1.00
    elif s_int in (16, 17):
        S = 1.03
    else:
        S = 1.06

    tv = round(p0 * G * M * S, 2)

    # เผื่อ Error 0.5% ของเซิร์ฟเวอร์ (กันตกเกรดหน้างาน)
    return {
        "TV": tv,
        "safe_g": g * 1.005,
        "safe_m": m * 1.005,
        "safe_s": s * 0.995,
    }


def generate_quota_proportional_orders(
    current_round: int,
    total_rounds: int,
    initial_total_bags: int,
    lots: List[Dict[str, Any]],
    routes: List[Dict[str, Any]],
    ceiling_memory: Dict[str, float] = None,
) -> List[Dict[str, Any]]:
    """
    อัลกอริทึม Quota_Proportional สำหรับลงแข่งจริง (อัปเกรดรองรับ Specialty + ขายได้ 1/4 ต่อรอบ + 9/16 รอบ)
    
    Parameters:
      - current_round: รอบปัจจุบัน (เริ่มที่ 1)
      - total_rounds: จำนวนรอบทั้งหมดของเกม (เช่น 9 หรือ 16)
      - initial_total_bags: จำนวนกระสอบรวมทั้งหมดที่ได้รับในรอบแรก
      - lots: รายการ Lot ของเรา เช่น [{"lot_id": "LOT-T25-1", "s": 17, "m": 11.5, "g": 2.1, "bags": 6}, ...]
      - routes: รายการผู้ซื้อในรอบนี้ เช่น [{"order_id": "R5-01", "buyer_type": "SPECIALTY", "quota_bags": 4, "max_defect": 3.0, "max_moisture": 12.5, "min_size": 17}, ...]
      - ceiling_memory: dict เก็บเพดานราคาสูงสุดที่เคยค้นพบของแต่ละบริษัท (ถ้ามี)
    """
    if ceiling_memory is None:
        ceiling_memory = {}

    # 1. คำนวณเพดานขายสูงสุดต่อรอบตามกติกาใหม่: floor(initial_bags / 4)
    max_sell_cap = max(1, initial_total_bags // 4)
    total_rem_bags = sum(int(l.get("bags", 0)) for l in lots)
    if total_rem_bags <= 0 or not routes:
        return []

    # ประเมินความเร่งด่วน (Urgency) เทียบกับจำนวนรอบที่เหลือและเพดาน 1/4 ต่อรอบ
    rounds_left = max(1, total_rounds - current_round + 1)
    min_rounds_needed = math.ceil(total_rem_bags / max_sell_cap)
    # ถ้าจำนวนรอบที่เหลือใกล้เคียงกับจำนวนรอบขั้นต่ำที่ต้องใช้ขาย จะเร่งสปีดทันที
    urgency = min(1.0, max(0.0, (current_round / total_rounds) * 0.6 + (min_rounds_needed / rounds_left) * 0.4))
    if rounds_left == 1:
        urgency = 1.0

    # 2. เตรียมข้อมูล Lot และคำนวณ TV + ค่าเผื่อ Error 0.5%
    eval_lots = []
    for l in lots:
        bags = int(l.get("bags", 0))
        if bags <= 0:
            continue
        info = calc_tv_and_safe_specs(float(l["s"]), float(l["m"]), float(l["g"]))
        eval_lots.append({**l, **info, "rem_bags": bags})

    # 3. จับคู่ Lot กับ Route ที่ผ่านสเปก (เช็กทั้ง defect, moisture, และ min_size สำหรับ Specialty)
    valid_routes_for_lot: Dict[str, List[Dict[str, Any]]] = {}
    for l in eval_lots:
        lid = l["lot_id"]
        valid_routes_for_lot[lid] = []
        for r in routes:
            q = int(r.get("quota_bags", 0))
            if q <= 0:
                continue
            b_type = str(r.get("buyer_type", "")).upper()
            max_g = float(r.get("max_defect", 3.0 if "SPEC" in b_type else (7.0 if "COMM" in b_type else 15.0)))
            max_m = float(r.get("max_moisture", 12.5 if ("SPEC" in b_type or "COMM" in b_type) else 14.0))
            min_s = float(r.get("min_size", 0.0))

            # เช็กแบบเผื่อ Error 0.5% ป้องกันโดนตีตก
            if l["safe_g"] <= max_g and l["safe_m"] <= max_m and l["safe_s"] >= min_s:
                valid_routes_for_lot[lid].append(r)

    # เรียงลำดับ Lot ที่ "ทางเลือกน้อยที่สุด" ไปจองโควตาก่อน (Opportunity-Cost)
    eval_lots.sort(key=lambda x: (len(valid_routes_for_lot[x["lot_id"]]), -x["TV"]))

    route_cap_left = {r["order_id"]: int(r["quota_bags"]) for r in routes}
    # เสนอขายเผื่อไว้ไม่เกิน 1.5 เท่าของ max_sell_cap เพื่อกันพลาดแต่ไม่กระจายราคาเสียเปล่า
    submit_budget = min(total_rem_bags, max(max_sell_cap + 1, int(math.ceil(max_sell_cap * 1.5))))
    submitted_total = 0
    orders: List[Dict[str, Any]] = []

    for l in eval_lots:
        if submitted_total >= submit_budget:
            break
        lid = l["lot_id"]
        avail_routes = [r for r in valid_routes_for_lot[lid] if route_cap_left[r["order_id"]] > 0]
        if not avail_routes:
            continue

        # จัดกลุ่มแยก Specialty ไว้ให้ Priority #1 (เพราะกำไร +22% ถึง +24% เหนือ TV)
        spec_routes = [r for r in avail_routes if "SPEC" in str(r.get("buyer_type", "")).upper()]
        normal_routes = [r for r in avail_routes if r not in spec_routes]

        alloc_plan: List[tuple] = []
        bags_to_use = min(l["rem_bags"], submit_budget - submitted_total)

        # 4A. ถ้ามีผู้ซื้อ Specialty ให้จัดสรรให้เต็มโควตาที่เปิดรับก่อนเสมอ
        for r in spec_routes:
            if bags_to_use <= 0:
                break
            oid = r["order_id"]
            take = min(bags_to_use, route_cap_left[oid])
            if take > 0:
                alloc_plan.append((r, take))
                bags_to_use -= take
                route_cap_left[oid] -= take

        # 4B. ส่วนที่เหลือ กระจายตามสัดส่วนโควตา (Quota-Proportional) ให้กลุ่ม Commercial / Instant
        if bags_to_use > 0 and normal_routes:
            total_q = sum(route_cap_left[r["order_id"]] for r in normal_routes)
            if total_q > 0:
                raw_shares = [(r, (route_cap_left[r["order_id"]] / total_q) * bags_to_use) for r in normal_routes]
                floors = [(r, int(math.floor(sh)), sh - math.floor(sh)) for r, sh in raw_shares]
                rem_bags = bags_to_use - sum(x[1] for x in floors)
                floors.sort(key=lambda x: x[2], reverse=True)

                for idx, (r, cnt, _) in enumerate(floors):
                    extra = 1 if idx < rem_bags else 0
                    oid = r["order_id"]
                    take = min(cnt + extra, route_cap_left[oid])
                    if take > 0:
                        alloc_plan.append((r, take))
                        route_cap_left[oid] -= take

        # 5. คำนวณราคาเสนอขาย (Sweet Spot Ratio + Micro Price Laddering)
        for r, qty in alloc_plan:
            oid = r["order_id"]
            b_type = str(r.get("buyer_type", "")).upper()
            comp_name = r.get("company_name", oid)

            # กำหนด Ratio ตั้งต้นตามกลุ่มผู้ซื้อ (เผื่อ error TV 0.008)
            if "SPEC" in b_type:
                default_top = 1.215 if float(r.get("min_size", 16)) >= 17 else 1.195
            elif "COMM" in b_type:
                default_top = 1.018
            else:
                default_top = 0.968

            known_ceil = ceiling_memory.get(comp_name)
            base_top = min(default_top, known_ceil - 0.008) if known_ceil else default_top

            # ลดราคาตาม Urgency เพื่อแย่งโควตาในตลาด 55 ทีม (80% demand)
            discount = 0.035 * urgency if "SPEC" not in b_type else 0.020 * urgency
            top_ratio = max(0.82, base_top - discount)

            for bag_i in range(qty):
                bid_ratio = max(0.81, round(top_ratio - (0.003 * bag_i), 4))
                bid_price = round(l["TV"] * bid_ratio, 2)
                orders.append({
                    "order_id": oid,
                    "lot_id": lid,
                    "bags": 1,
                    "price": bid_price,
                    "ratio": bid_ratio,
                })
                submitted_total += 1

    return orders