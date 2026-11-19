import math
from typing import Any, Dict, List, Optional


GRADE_TABLE = [
    (1.0, "PREMIUM", 1.08),
    (3.0, "GRADE_1", 1.04),
    (5.0, "GRADE_2", 1.00),
    (7.0, "GRADE_3", 0.96),
    (15.0, "BELOW_STANDARD", 0.85),
    (float("inf"), "REJECT", 0.60),
]


def calc_tv(s: float, m: float, g: float, p0: float = 120.0) -> float:
    """คำนวณราคาประเมินกลาง (TV = 120 * G * M * S)"""
    grade_mult = 0.60
    for max_d, _, g_mult in GRADE_TABLE:
        if g <= max_d:
            grade_mult = g_mult
            break
    G = round((1.0 - g / 100.0) * grade_mult, 4)

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

    return round(p0 * G * M * S, 2)


class QuotaPropV4LiveEngine:
    """
    สมองบอทแข่งจริง TESA Top Gun Rally 2026 (Quota_Prop_V4):
      1. คัดสเปกเผื่อ Error 0.5% (safe_g = g*1.005, safe_m = m*1.005) ป้องกันโดนตีตกหน้างาน
      2. ดักจับผู้ซื้อ SPECIALTY (บางกอกบลูม R5-01 เพดาน ~1.24 / ดอยช้าง R5-02 เพดาน ~1.22)
         โดยหยอดล็อตผ่านเกณฑ์แค่ 1-2 ใบต่อเจ้าเพื่อฟันกำไร +22% โดยไม่เอาของไปกองตาย
      3. กระจายกระสอบที่เหลือด้วยลูป while (คำนวณ tot_q ใหม่ทุกขั้น) ให้ส่งออกครบ 100% ของล็อต
      4. คุมจังหวะเร่งระบายตามเพดานขายต่อรอบ floor(initial_bags / 4) รองรับทั้งโหมด 9 รอบ และ 16 รอบ
    """

    def __init__(self, default_total_rounds: int = 9):
        self.default_total_rounds = default_total_rounds
        self.initial_total_bags: int = 0
        self.ceiling_memory: Dict[str, float] = {}
        self.last_tv_map: Dict[str, float] = {}
        self.last_route_company_map: Dict[str, str] = {}

    def reset_game(self):
        self.initial_total_bags = 0
        self.ceiling_memory.clear()
        self.last_tv_map.clear()
        self.last_route_company_map.clear()

    def _parse_lot(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        lot_id = str(raw.get("lot_id") or raw.get("id") or raw.get("lotId") or "")
        s = float(raw.get("s") or raw.get("screen_size") or raw.get("size") or raw.get("screen") or 15.0)
        m = float(raw.get("m") or raw.get("moisture") or 12.0)
        g = float(raw.get("g") or raw.get("defect") or raw.get("defective") or 5.0)
        bags = int(raw.get("bags") or raw.get("quantity") or raw.get("remaining_bags") or raw.get("amount") or 0)
        tv = float(raw.get("tv") or raw.get("TV") or calc_tv(s, m, g))
        return {
            "lot_id": lot_id,
            "s": s,
            "m": m,
            "g": g,
            "bags": bags,
            "TV": tv,
        }

    def _parse_route(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        order_id = str(
            raw.get("order_id") or raw.get("rfq_id") or raw.get("route_id") or raw.get("id") or ""
        )
        company_name = str(
            raw.get("company_name") or raw.get("buyer_name") or raw.get("name") or order_id
        )
        raw_type = str(raw.get("buyer_type") or raw.get("type") or raw.get("category") or "").upper()
        quota_bags = int(
            raw.get("quota_bags") or raw.get("quota") or raw.get("demand") or raw.get("quantity") or 0
        )

        # ตรวจจับประเภทผู้ซื้ออัตโนมัติ (SPECIALTY / COMMERCIAL / INSTANT)
        if "SPEC" in raw_type or "บางกอกบลูม" in company_name or "ดอยช้าง" in company_name:
            buyer_type = "SPECIALTY"
        elif "COMM" in raw_type or any(k in company_name for k in ("พรีเมียม", "โกลบอล", "สยามคอม", "ชุมพร")):
            buyer_type = "COMMERCIAL"
        elif "INST" in raw_type:
            buyer_type = "INSTANT"
        else:
            # เดาจากเกณฑ์ defect ถ้าไม่ได้ระบุชื่อชัดเจน
            max_d_hint = float(raw.get("max_defect") or raw.get("defect_max") or 15.0)
            if max_d_hint <= 3.5:
                buyer_type = "SPECIALTY"
            elif max_d_hint <= 7.5:
                buyer_type = "COMMERCIAL"
            else:
                buyer_type = "INSTANT"

        default_def = 3.0 if buyer_type == "SPECIALTY" else (7.0 if buyer_type == "COMMERCIAL" else 15.0)
        default_moist = 12.5 if buyer_type in ("SPECIALTY", "COMMERCIAL") else 14.0
        default_min_s = 17.0 if "บางกอกบลูม" in company_name else (16.0 if buyer_type == "SPECIALTY" else 0.0)

        max_defect = float(raw.get("max_defect") or raw.get("defect_max") or default_def)
        max_moisture = float(raw.get("max_moisture") or raw.get("moisture_max") or default_moist)
        min_size = float(raw.get("min_size") or raw.get("min_screen") or raw.get("screen_min") or default_min_s)

        return {
            "order_id": order_id,
            "company_name": company_name,
            "buyer_type": buyer_type,
            "quota_bags": quota_bags,
            "max_defect": max_defect,
            "max_moisture": max_moisture,
            "min_size": min_size,
        }

    def observe_round_feedback(self, feedback_list: List[Dict[str, Any]]):
        """เรียนรู้เพดานราคาผู้ซื้ออัตโนมัติเมื่อมีออเดอร์ถูกตีกลับว่า 'เกินเพดาน'"""
        if not feedback_list:
            return
        for fb in feedback_list:
            oid = str(fb.get("order_id") or fb.get("rfq_id") or fb.get("rfq") or fb.get("route_id") or "")
            lid = str(fb.get("lot_id") or fb.get("lot") or "")
            status = str(fb.get("status") or fb.get("reason") or fb.get("message") or "").lower()
            price = float(fb.get("price") or fb.get("bid_price") or 0.0)

            if any(k in status for k in ("เกินเพดาน", "ceiling", "too high", "expensive")) and price > 0:
                comp_name = self.last_route_company_map.get(oid, oid)
                tv = self.last_tv_map.get(lid, 0.0)
                if tv > 0:
                    failed_ratio = price / tv
                    prev = self.ceiling_memory.get(comp_name, 1.35)
                    self.ceiling_memory[comp_name] = min(prev, round(failed_ratio - 0.002, 4))

    def compute_round_orders(
        self,
        current_round: int,
        raw_lots: List[Dict[str, Any]],
        raw_routes: List[Dict[str, Any]],
        total_rounds: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        สร้างคำสั่งเสนอขายสำหรับรอบปัจจุบัน:
        คืนค่าเป็น List ของ Dict ที่มีคีย์ครบทั้งรูปแบบรายกระสอบและแบบรวมล็อต
        """
        t_rounds = total_rounds if (total_rounds and total_rounds > 0) else self.default_total_rounds
        lots = [self._parse_lot(x) for x in raw_lots]
        routes = [self._parse_route(x) for x in raw_routes]

        active_lots = [l for l in lots if l["bags"] > 0]
        total_rem = sum(l["bags"] for l in active_lots)
        if current_round <= 1 or self.initial_total_bags == 0:
            self.initial_total_bags = max(total_rem, self.initial_total_bags)

        if total_rem <= 0 or not routes:
            return []

        for l in active_lots:
            self.last_tv_map[l["lot_id"]] = l["TV"]
        for r in routes:
            self.last_route_company_map[r["order_id"]] = r["company_name"]

        # กติกาใหม่: ขายได้สูงสุดต่อรอบ = floor(initial_bags / 4)
        max_sell_cap = max(1, self.initial_total_bags // 4)
        rounds_left = max(1, t_rounds - current_round + 1)
        cap_ratio = min(1.0, math.ceil(total_rem / max_sell_cap) / rounds_left)
        urgency = max(current_round / t_rounds, cap_ratio)

        cap_left = {r["order_id"]: r["quota_bags"] for r in routes}
        orders: List[Dict[str, Any]] = []

        # เรียงล็อตจาก TV ต่ำไปสูง (ให้ของเกรดรองจองโควตา Instant ก่อน ของเกรดสูงเก็บเข้า Specialty/Commercial)
        for l in sorted(active_lots, key=lambda x: x["TV"]):
            safe_g = l["g"] * 1.005
            safe_m = l["m"] * 1.005

            avail = [
                r for r in routes
                if cap_left[r["order_id"]] > 0
                and safe_g <= r["max_defect"]
                and safe_m <= r["max_moisture"]
                and l["s"] >= r["min_size"]
            ]
            if not avail:
                continue

            rem_bags = l["bags"]
            alloc = []
            rem_routes = sorted(
                avail,
                key=lambda r: (
                    0 if r["buyer_type"] == "SPECIALTY" else (1 if r["buyer_type"] == "COMMERCIAL" else 2),
                    -cap_left[r["order_id"]],
                ),
            )

            # คำนวณ tot_q ใหม่ทุกขั้นในลูป while เพื่อให้จัดสรรกระสอบออกครบ 100% ของล็อต
            while rem_bags > 0 and rem_routes:
                tot_q = sum(cap_left[r["order_id"]] for r in rem_routes)
                if tot_q <= 0:
                    break
                r = rem_routes.pop(0)
                oid = r["order_id"]
                share = max(1, int(round((cap_left[oid] / tot_q) * rem_bags)))
                if r["buyer_type"] == "SPECIALTY":
                    share = min(share, 1 if r["quota_bags"] <= 3 else 2)
                take = min(share, rem_bags, cap_left[oid])
                if take > 0:
                    alloc.append((r, take))
                    rem_bags -= take
                    cap_left[oid] -= take

            for r, qty in alloc:
                oid = r["order_id"]
                comp_name = r["company_name"]
                b_type = r["buyer_type"]

                if b_type == "SPECIALTY":
                    base_r = 1.215 if r["min_size"] >= 17 else 1.200
                elif b_type == "COMMERCIAL":
                    base_r = 1.014
                else:
                    base_r = 0.964

                known = self.ceiling_memory.get(comp_name)
                if known:
                    base_r = min(base_r, known - 0.004)

                extra_cut = 0.008 if cap_ratio >= 0.85 else 0.0015
                ratio = max(0.815, base_r - 0.04 * urgency - extra_cut)

                for i in range(qty):
                    bid_ratio = round(ratio - 0.002 * i, 4)
                    bid_price = round(l["TV"] * bid_ratio, 2)
                    orders.append({
                        "order_id": oid,
                        "rfq_id": oid,
                        "route_id": oid,
                        "buyer_name": comp_name,
                        "lot_id": l["lot_id"],
                        "bags": 1,
                        "quantity": 1,
                        "price": bid_price,
                        "bid_price": bid_price,
                        "target_ratio": bid_ratio,
                        "tv": l["TV"],
                    })

        return orders