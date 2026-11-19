from typing import List, Dict
import numpy as np
from coffee_math import CoffeeLot, BuyerRoute
from action_mask import get_valid_route_mask


class BagCombinatorialRouter:
    def __init__(self, error_buffer: float = 0.005):
        self.error_buffer = error_buffer

    def _is_specialty_lot(self, lot: CoffeeLot) -> bool:
        safe_s = lot.s * (1.0 - self.error_buffer)
        safe_g = lot.g * (1.0 + self.error_buffer)
        safe_m = lot.m * (1.0 + self.error_buffer)
        return (safe_s >= 16.0 and safe_g <= 3.0 and safe_m <= 12.5)

    def route_bags(
        self,
        lots: List[CoffeeLot],
        routes: List[BuyerRoute],
        target_sell_bags: int,
        price_multiplier: float,
        estimated_ceilings: Dict[str, float],
        inst_price_multiplier: float = None,
        protect_spec: bool = False,
        max_spike_reserve_bags: int = 18,
    ) -> List[Dict]:
        """
        จัดสรรกระสอบลงแต่ละ Route อย่างชาญฉลาด:
        1. Priority #1 (SPECIALTY): กระจายแบบ Round-Robin ลงทุก RFQ ของ Specialty ที่เปิดรับ
           พร้อม Price Laddering ชิดเพดานจริง เพื่อลดการกระจุกตัวและเพิ่มโอกาสติดโควตาสูงสุด
        2. Priority #2 (COMMERCIAL & INSTANT): กระจายกระสอบที่เหลือเข้า Commercial ก่อน Instant
           (หาก protect_spec=True จะสงวนเมล็ด Specialty ไว้ไม่เกิน max_spike_reserve_bags สำหรับช่วง Spike)
        """
        if target_sell_bags <= 0 or not lots or not routes:
            return []

        if inst_price_multiplier is None:
            inst_price_multiplier = price_multiplier * 0.96

        valid_mask = get_valid_route_mask(lots, routes, self.error_buffer)
        rem_lot_bags = [lot.bags for lot in lots]
        rem_route_quota = [r.quota for r in routes]
        rem_to_sell = min(target_sell_bags, sum(rem_lot_bags))

        orders: List[Dict] = []

        # --- PHASE 1: กระจายกระสอบลงทุก RFQ ของ SPECIALTY แบบ Round-Robin + Price Laddering ---
        spec_indices = [
            j for j, r in enumerate(routes)
            if r.buyer_type == "SPECIALTY" and rem_route_quota[j] > 0
        ]
        spec_indices.sort(
            key=lambda j: (
                estimated_ceilings.get(routes[j].route_id, routes[j].base_ceiling_mult),
                routes[j].min_screen,
            ),
            reverse=True,
        )

        spec_step_counter: Dict[str, int] = {}
        while rem_to_sell > 0 and spec_indices:
            progressed_spec = False
            for j in spec_indices:
                if rem_to_sell <= 0 or rem_route_quota[j] <= 0:
                    continue
                route = routes[j]
                min_req = max(1, getattr(route, "min_bags_per_bid", 1))
                if rem_route_quota[j] < min_req or rem_to_sell < min_req:
                    continue

                eligible_lots = [
                    i for i in range(len(lots))
                    if valid_mask[i, j] and rem_lot_bags[i] >= min_req
                ]
                if not eligible_lots:
                    continue
                eligible_lots.sort(key=lambda i: lots[i].tv, reverse=True)

                i = eligible_lots[0]
                chunk = min(max(min_req, 2), rem_lot_bags[i], rem_route_quota[j], rem_to_sell)
                if chunk >= min_req:
                    ceil_mult = estimated_ceilings.get(route.route_id, route.base_ceiling_mult)
                    aggr_scale = float(np.clip(0.982 + (price_multiplier - 0.91) * 0.14, 0.965, 1.0))
                    top_mult = min(ceil_mult - 0.004, max(1.15, ceil_mult * aggr_scale))

                    step_idx = spec_step_counter.get(route.route_id, 0)
                    bid_mult = max(1.12, round(top_mult - 0.002 * step_idx, 4))
                    spec_step_counter[route.route_id] = step_idx + 1

                    orders.append({
                        "lot_id": lots[i].lot_id,
                        "lot": lots[i],
                        "route_id": route.route_id,
                        "qty": chunk,
                        "price": round(lots[i].tv * bid_mult, 2),
                        "buyer_type": "SPECIALTY",
                    })
                    rem_lot_bags[i] -= chunk
                    rem_route_quota[j] -= chunk
                    rem_to_sell -= chunk
                    progressed_spec = True

            if not progressed_spec:
                break

        # --- PHASE 2: กระจายกระสอบที่เหลือให้ COMMERCIAL และ INSTANT ---
        non_spec_indices = [
            j for j, r in enumerate(routes)
            if r.buyer_type != "SPECIALTY" and rem_route_quota[j] > 0
        ]
        non_spec_indices.sort(
            key=lambda j: (0 if routes[j].buyer_type == "COMMERCIAL" else 1, -routes[j].quota)
        )

        while rem_to_sell > 0:
            progressed = False
            current_spec_rem = sum(
                rem_lot_bags[idx] for idx, l in enumerate(lots) if self._is_specialty_lot(l)
            )
            for j in non_spec_indices:
                if rem_to_sell <= 0 or rem_route_quota[j] <= 0:
                    continue
                route = routes[j]
                min_req = max(1, getattr(route, "min_bags_per_bid", 1))
                if rem_route_quota[j] < min_req or rem_to_sell < min_req:
                    continue

                eligible_lots = [
                    i for i in range(len(lots))
                    if valid_mask[i, j] and rem_lot_bags[i] >= min_req
                    and not (
                        protect_spec
                        and self._is_specialty_lot(lots[i])
                        and current_spec_rem <= max_spike_reserve_bags
                    )
                ]
                if not eligible_lots:
                    continue

                if route.buyer_type == "INSTANT":
                    eligible_lots.sort(key=lambda i: lots[i].tv)
                else:
                    eligible_lots.sort(key=lambda i: lots[i].tv, reverse=True)

                i = eligible_lots[0]
                step_qty = min(max(min_req, 2), rem_lot_bags[i], rem_route_quota[j], rem_to_sell)
                if step_qty >= min_req:
                    ceil_mult = estimated_ceilings.get(route.route_id, route.base_ceiling_mult)
                    raw_mult = price_multiplier if route.buyer_type == "COMMERCIAL" else inst_price_multiplier
                    safe_mult = min(ceil_mult - 0.003, max(0.82, raw_mult))

                    existing = next(
                        (o for o in orders if o["lot_id"] == lots[i].lot_id and o["route_id"] == route.route_id),
                        None
                    )
                    if existing:
                        existing["qty"] += step_qty
                    else:
                        orders.append({
                            "lot_id": lots[i].lot_id,
                            "lot": lots[i],
                            "route_id": route.route_id,
                            "qty": step_qty,
                            "price": round(lots[i].tv * safe_mult, 2),
                            "buyer_type": route.buyer_type
                        })

                    rem_lot_bags[i] -= step_qty
                    rem_route_quota[j] -= step_qty
                    rem_to_sell -= step_qty
                    progressed = True

            if not progressed:
                break

        return orders