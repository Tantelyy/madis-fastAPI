#!/usr/bin/env python3
"""Generate coherent historical inventory data for the existing MADIS products.

This script deliberately creates no Product or product referential.  It creates
dedicated, dated inventory lots so the simulated history neither consumes nor
rewrites the real lots already present in a development database.
"""

from __future__ import annotations

import argparse
import math
import os
import random
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Iterable, Sequence
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from dotenv import load_dotenv
import psycopg


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_VERSION = "v1"
MARKER_PREFIX = "[SYNTHETIC_DEMAND"


@dataclass(frozen=True)
class Config:
    start_date: date
    end_date: date
    random_seed: int
    promotion_probability: float
    refund_probability: float
    cancellation_probability: float
    adjustment_probability: float
    reorder_threshold_days: int
    min_promotion_days: int
    max_promotion_days: int
    actor_id: int | None
    supplier_id: int | None
    default_purchase_price: Decimal

    @property
    def days(self) -> int:
        return (self.end_date - self.start_date).days + 1

    @property
    def batch_key(self) -> str:
        # The period is intentionally part of the key.  A different seed is
        # still refused for the same period: overlapping synthetic histories
        # would not be safe to merge without a dedicated batch column/table.
        return f"{SCRIPT_VERSION}|{self.start_date.isoformat()}|{self.end_date.isoformat()}"

    @property
    def marker(self) -> str:
        return f"{MARKER_PREFIX}|{self.batch_key}|seed={self.random_seed}]"


@dataclass(frozen=True)
class Product:
    id: int
    name: str
    reference: str
    purchase_price: Decimal
    sale_price: Decimal
    wholesale_price: Decimal


@dataclass(frozen=True)
class ExcludedProduct:
    id: int
    reference: str
    name: str
    reason: str


@dataclass(frozen=True)
class ProductLoadReport:
    products: list[Product]
    total_found: int
    excluded_products: list[ExcludedProduct]


@dataclass(frozen=True)
class ProductProfile:
    name: str
    base_demand: float
    trend_per_year: float
    volatility: float
    zero_probability: float
    promotion_sensitivity: float
    seasonality_amplitude: float
    seasonality_phase: float
    weekday_factors: tuple[float, ...]


@dataclass
class Lot:
    local_id: int
    product_id: int
    created_at: datetime
    quantity: int
    remaining_quantity: int
    purchase_price: Decimal
    sale_price: Decimal
    wholesale_price: Decimal
    adjustment_delta: int = 0
    last_changed_at: datetime | None = None
    database_id: int | None = None


@dataclass(frozen=True)
class Promotion:
    local_id: int
    source_product_id: int
    gift_product_id: int
    start_at: datetime
    end_at: datetime
    buy_quantity: int
    free_quantity: int


@dataclass
class CartLine:
    lot: Lot
    paid_quantity: int
    free_quantity: int
    promotion: Promotion | None
    database_id: int | None = None


@dataclass
class CartPlan:
    local_id: int
    created_at: datetime
    status: str
    lines: list[CartLine]
    total_price: Decimal
    reason: str
    paid_at: datetime | None
    database_id: int | None = None


@dataclass(frozen=True)
class Movement:
    lot: Lot
    incoming_quantity: int
    outgoing_quantity: int
    movement_type: str
    created_at: datetime
    cart: CartPlan | None = None


@dataclass(frozen=True)
class Refund:
    line: CartLine
    cart: CartPlan
    quantity: int
    created_at: datetime


@dataclass
class Simulation:
    lots: list[Lot] = field(default_factory=list)
    promotions: list[Promotion] = field(default_factory=list)
    carts: list[CartPlan] = field(default_factory=list)
    movements: list[Movement] = field(default_factory=list)
    refunds: list[Refund] = field(default_factory=list)
    daily_demand: dict[tuple[date, int], int] = field(
        default_factory=lambda: defaultdict(int),
    )
    daily_sales: dict[tuple[date, int], int] = field(
        default_factory=lambda: defaultdict(int),
    )


PROFILE_TEMPLATES: tuple[tuple[str, float, float, float, float, float], ...] = (
    ("VERY_HIGH_ROTATION", 18.0, 0.20, 0.02, 0.10, 0.10),
    ("HIGH_ROTATION", 10.0, 0.22, 0.04, 0.12, 0.08),
    ("MEDIUM_ROTATION", 5.0, 0.28, 0.10, 0.18, 0.05),
    ("LOW_ROTATION", 2.0, 0.38, 0.35, 0.20, 0.00),
    ("VERY_LOW_ROTATION", 0.55, 0.52, 0.58, 0.15, 0.00),
    ("IRREGULAR", 3.5, 0.72, 0.30, 0.35, 0.00),
    ("PROMOTION_SENSITIVE", 4.5, 0.28, 0.14, 0.85, 0.03),
    ("STABLE", 4.0, 0.12, 0.02, 0.08, 0.00),
    ("GROWING", 4.0, 0.25, 0.16, 0.25, 0.13),
    ("DECLINING", 4.0, 0.25, 0.16, 0.20, -0.13),
)


def env_value(name: str, default: str, legacy_name: str | None = None) -> str:
    return os.getenv(name) or (os.getenv(legacy_name) if legacy_name else None) or default


def parse_date(value: str, variable_name: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as error:
        raise ValueError(f"{variable_name} doit respecter le format YYYY-MM-DD.") from error


def parse_probability(value: str, variable_name: str) -> float:
    try:
        parsed = float(value)
    except ValueError as error:
        raise ValueError(f"{variable_name} doit être un nombre entre 0 et 1.") from error
    if not 0 <= parsed <= 1:
        raise ValueError(f"{variable_name} doit être compris entre 0 et 1.")
    return parsed


def optional_positive_int(value: str | None, variable_name: str) -> int | None:
    if not value:
        return None
    parsed = int(value)
    if parsed < 1:
        raise ValueError(f"{variable_name} doit être supérieur à zéro.")
    return parsed


def load_config(arguments: argparse.Namespace) -> Config:
    default_end = date.today() - timedelta(days=1)
    default_start = default_end - timedelta(days=364)
    start = parse_date(
        arguments.start_date
        or env_value("SYNTHETIC_START_DATE", default_start.isoformat(), "START_DATE"),
        "SYNTHETIC_START_DATE/START_DATE",
    )
    end = parse_date(
        arguments.end_date
        or env_value("SYNTHETIC_END_DATE", default_end.isoformat(), "END_DATE"),
        "SYNTHETIC_END_DATE/END_DATE",
    )
    if end < start:
        raise ValueError("La date de fin doit être postérieure ou égale à la date de début.")

    minimum = int(env_value("SYNTHETIC_MIN_PROMOTION_DAYS", "2", "MIN_PROMOTION_DAYS"))
    maximum = int(env_value("SYNTHETIC_MAX_PROMOTION_DAYS", "7", "MAX_PROMOTION_DAYS"))
    if minimum < 1 or maximum < minimum:
        raise ValueError("Les durées de promotion sont invalides.")

    return Config(
        start_date=start,
        end_date=end,
        random_seed=int(
            arguments.seed or env_value("SYNTHETIC_RANDOM_SEED", "42", "RANDOM_SEED"),
        ),
        promotion_probability=parse_probability(
            env_value("SYNTHETIC_PROMOTION_PROBABILITY", "0.003", "PROMOTION_PROBABILITY"),
            "SYNTHETIC_PROMOTION_PROBABILITY/PROMOTION_PROBABILITY",
        ),
        refund_probability=parse_probability(
            env_value("SYNTHETIC_REFUND_PROBABILITY", "0.007", "REFUND_PROBABILITY"),
            "SYNTHETIC_REFUND_PROBABILITY/REFUND_PROBABILITY",
        ),
        cancellation_probability=parse_probability(
            env_value("SYNTHETIC_CANCELLATION_PROBABILITY", "0.0015", "CANCELLATION_PROBABILITY"),
            "SYNTHETIC_CANCELLATION_PROBABILITY/CANCELLATION_PROBABILITY",
        ),
        adjustment_probability=parse_probability(
            env_value("SYNTHETIC_ADJUSTMENT_PROBABILITY", "0.0008", "ADJUSTMENT_PROBABILITY"),
            "SYNTHETIC_ADJUSTMENT_PROBABILITY/ADJUSTMENT_PROBABILITY",
        ),
        reorder_threshold_days=int(
            env_value("SYNTHETIC_REORDER_THRESHOLD_DAYS", "14", "REORDER_THRESHOLD"),
        ),
        min_promotion_days=minimum,
        max_promotion_days=maximum,
        actor_id=optional_positive_int(os.getenv("SYNTHETIC_ACTOR_ID"), "SYNTHETIC_ACTOR_ID"),
        supplier_id=optional_positive_int(
            os.getenv("SYNTHETIC_SUPPLIER_ID"),
            "SYNTHETIC_SUPPLIER_ID",
        ),
        default_purchase_price=Decimal(
            env_value("SYNTHETIC_DEFAULT_PURCHASE_PRICE", "5000"),
        ).quantize(Decimal("0.01")),
    )


def ensure_not_production() -> None:
    production_variables = ("APP_ENV", "ENVIRONMENT", "NODE_ENV")
    active = [
        name
        for name in production_variables
        if os.getenv(name, "").strip().lower() == "production"
    ]
    if active:
        raise RuntimeError(
            "Génération refusée : environnement de production détecté via "
            + ", ".join(active)
            + ".",
        )


def postgres_connection_settings(database_url: str) -> tuple[str, str | None]:
    """Translate Prisma's optional `schema` URL parameter for psycopg.

    Prisma accepts ``?schema=public`` whereas libpq does not.  Keeping the URL
    otherwise unchanged still makes DATABASE_URL the sole connection source.
    """
    parsed = urlsplit(database_url)
    query = parse_qsl(parsed.query, keep_blank_values=True)
    schema = next((value for key, value in query if key == "schema"), None)
    filtered_query = [(key, value) for key, value in query if key != "schema"]
    if schema and not schema.replace("_", "").isalnum():
        raise ValueError("Le paramètre schema de DATABASE_URL est invalide.")
    normalized_url = urlunsplit(
        (parsed.scheme, parsed.netloc, parsed.path, urlencode(filtered_query), parsed.fragment),
    )
    return normalized_url, f"-c search_path={schema}" if schema else None


def at(day: date, hour: int, minute: int = 0) -> datetime:
    return datetime.combine(day, time(hour, minute))


def random_business_time(rng: random.Random, day: date) -> datetime:
    return at(day, rng.randint(8, 19), rng.randint(0, 59))


def load_existing_products(
    connection: psycopg.Connection,
    config: Config,
) -> ProductLoadReport:
    """Load active Products only, reusing a latest lot price where available."""
    fallback_purchase = config.default_purchase_price
    with connection.cursor() as cursor:
        cursor.execute(
            '''
            SELECT "retailAverage", "wholeSaleAverage"
            FROM "PricingRules" rules
            JOIN "PricingGrids" grids ON grids."ID" = rules."pricingGridId"
            WHERE grids.status = 'ACTIVE'
              AND grids."effectiveFrom" <= CURRENT_TIMESTAMP
              AND (grids."effectiveTo" IS NULL OR grids."effectiveTo" >= CURRENT_TIMESTAMP)
              AND rules."minPurchasePrice" <= %s
              AND rules."maxPurchasePrice" >= %s
            ORDER BY rules."minPurchasePrice" DESC
            LIMIT 1
            ''',
            (fallback_purchase, fallback_purchase),
        )
        pricing_rule = cursor.fetchone()
        cursor.execute(
            '''
            SELECT p."ID", p.name, p.reference, p."deletedAt",
                   latest."purchasePrice", latest."salePrice", latest."wholeSalePrice"
            FROM "Products" p
            LEFT JOIN LATERAL (
              SELECT i."purchasePrice", i."salePrice", i."wholeSalePrice"
              FROM "Inventories" i
              WHERE i."productId" = p."ID"
              ORDER BY i."createdAt" DESC, i."ID" DESC
              LIMIT 1
            ) latest ON TRUE
            ORDER BY p."ID" ASC
            ''',
        )
        rows = cursor.fetchall()

    if not rows:
        return ProductLoadReport([], 0, [])
    active_rows = [row for row in rows if row[3] is None]
    if not pricing_rule and any(row[4] is None for row in active_rows):
        raise RuntimeError(
            "Au moins un produit n'a pas de lot existant et aucune grille de prix active "
            "ne permet de créer son lot synthétique.",
        )

    products: list[Product] = []
    excluded_products: list[ExcludedProduct] = []
    for product_id, name, reference, deleted_at, purchase, sale, wholesale in rows:
        if deleted_at is not None:
            excluded_products.append(
                ExcludedProduct(
                    product_id,
                    reference,
                    name,
                    "produit désactivé (deletedAt renseigné)",
                ),
            )
            continue
        if purchase is None:
            retail_margin, wholesale_margin = pricing_rule
            purchase = fallback_purchase
            sale = Decimal(round(float(purchase + retail_margin))).quantize(Decimal("0.01"))
            wholesale = Decimal(round(float(purchase + wholesale_margin))).quantize(Decimal("0.01"))
        products.append(
            Product(product_id, name, reference, purchase, sale, wholesale),
        )
    return ProductLoadReport(products, len(rows), excluded_products)


def load_actor_and_supplier(
    connection: psycopg.Connection,
    config: Config,
) -> tuple[int, int]:
    with connection.cursor() as cursor:
        cursor.execute(
            '''SELECT "ID" FROM "Users"
               WHERE "deletedAt" IS NULL
                 AND (%s::integer IS NULL OR "ID" = %s)
               ORDER BY "ID" ASC LIMIT 1''',
            (config.actor_id, config.actor_id),
        )
        actor = cursor.fetchone()
        cursor.execute(
            '''SELECT "ID" FROM "Suppliers"
               WHERE "deletedAt" IS NULL
                 AND (%s::integer IS NULL OR "ID" = %s)
               ORDER BY "ID" ASC LIMIT 1''',
            (config.supplier_id, config.supplier_id),
        )
        supplier = cursor.fetchone()
    if not actor:
        raise RuntimeError("Aucun utilisateur actif exploitable pour actorId.")
    if not supplier:
        raise RuntimeError(
            "Aucun fournisseur actif exploitable : nécessaire pour créer des lots synthétiques.",
        )
    return actor[0], supplier[0]


def existing_batch_count(connection: psycopg.Connection, config: Config) -> int:
    with connection.cursor() as cursor:
        cursor.execute(
            'SELECT count(*) FROM "Carts" WHERE reason LIKE %s',
            (f"{MARKER_PREFIX}|{config.batch_key}|%",),
        )
        return int(cursor.fetchone()[0])


def build_weekday_factors(rng: random.Random, profile_name: str) -> tuple[float, ...]:
    stable = profile_name == "STABLE"
    weekend_bias = rng.uniform(-0.05, 0.22)
    base = (1.0, 0.98, 1.01, 1.02, 1.08, 1.0 + weekend_bias, 0.96 + weekend_bias)
    return tuple(
        max(0.75, min(1.35, factor + rng.uniform(-0.025, 0.025 if not stable else 0.01)))
        for factor in base
    )


def build_product_profiles(products: Sequence[Product], rng: random.Random) -> dict[int, ProductProfile]:
    templates = list(PROFILE_TEMPLATES)
    rng.shuffle(templates)
    profiles: dict[int, ProductProfile] = {}
    for index, product in enumerate(products):
        name, base, volatility, zero_probability, sensitivity, trend = templates[index % len(templates)]
        profiles[product.id] = ProductProfile(
            name=name,
            base_demand=base * rng.uniform(0.82, 1.18),
            trend_per_year=trend * rng.uniform(0.75, 1.25),
            volatility=volatility,
            zero_probability=zero_probability,
            promotion_sensitivity=sensitivity,
            seasonality_amplitude=rng.uniform(0.05, 0.17),
            seasonality_phase=rng.uniform(0, math.tau),
            weekday_factors=build_weekday_factors(rng, name),
        )
    return profiles


def generate_promotions(
    products: Sequence[Product],
    profiles: dict[int, ProductProfile],
    config: Config,
    rng: random.Random,
) -> list[Promotion]:
    promotions: list[Promotion] = []
    active_until: dict[int, date] = {}
    current = config.start_date
    local_id = 1
    while current <= config.end_date:
        for product in products:
            if current <= active_until.get(product.id, config.start_date - timedelta(days=1)):
                continue
            profile = profiles[product.id]
            chance = config.promotion_probability * (0.45 + profile.promotion_sensitivity)
            if rng.random() >= chance:
                continue
            duration = rng.randint(config.min_promotion_days, config.max_promotion_days)
            end = min(config.end_date, current + timedelta(days=duration - 1))
            candidates = [candidate for candidate in products if candidate.id != product.id]
            if not candidates:
                continue
            gift = rng.choice(candidates)
            promotions.append(
                Promotion(
                    local_id=local_id,
                    source_product_id=product.id,
                    gift_product_id=gift.id,
                    start_at=at(current, 0),
                    end_at=at(end, 23, 59),
                    buy_quantity=rng.choice((2, 2, 3)),
                    free_quantity=1,
                ),
            )
            active_until[product.id] = end
            local_id += 1
        current += timedelta(days=1)
    return promotions


def active_promotion(
    promotions_by_product: dict[int, list[Promotion]],
    product_id: int,
    timestamp: datetime,
) -> Promotion | None:
    return next(
        (
            promotion
            for promotion in promotions_by_product.get(product_id, [])
            if promotion.start_at <= timestamp <= promotion.end_at
        ),
        None,
    )


def calculate_expected_demand(
    profile: ProductProfile,
    day: date,
    day_index: int,
    total_days: int,
    promotion: Promotion | None,
    rng: random.Random,
) -> int:
    weekday_factor = profile.weekday_factors[day.weekday()]
    seasonality = 1 + profile.seasonality_amplitude * math.sin(
        math.tau * day.timetuple().tm_yday / 365.25 + profile.seasonality_phase,
    )
    trend = 1 + profile.trend_per_year * (day_index / max(total_days - 1, 1))
    promotion_factor = 1 + (profile.promotion_sensitivity * 0.80 if promotion else 0)
    expected = profile.base_demand * weekday_factor * seasonality * trend * promotion_factor
    if rng.random() < profile.zero_probability:
        return 0
    noise = rng.gauss(0, max(0.35, math.sqrt(expected) * profile.volatility))
    return max(0, int(round(expected + noise)))


def split_transactions(quantity: int, rng: random.Random, minimum_chunk: int = 1) -> list[int]:
    if quantity <= 0:
        return []
    transaction_count = min(
        max(1, math.ceil(quantity / 4)),
        rng.randint(1, min(4, quantity)),
    )
    chunks: list[int] = []
    remaining = quantity
    for position in range(transaction_count - 1):
        minimum_remaining = transaction_count - position - 1
        upper = max(1, remaining - minimum_remaining)
        chunk = rng.randint(1, upper)
        chunks.append(chunk)
        remaining -= chunk
    chunks.append(remaining)
    # The total always remains exact. A promotion can still be applied to a
    # transaction whose quantity reaches its buy threshold.
    return [chunk for chunk in chunks if chunk >= minimum_chunk] or [quantity]


def allocate_stock(lots: Sequence[Lot], quantity: int) -> list[tuple[Lot, int]]:
    allocations: list[tuple[Lot, int]] = []
    remaining = quantity
    for lot in sorted(lots, key=lambda item: (item.created_at, item.local_id)):
        if remaining == 0:
            break
        allocated = min(lot.remaining_quantity, remaining)
        if allocated <= 0:
            continue
        lot.remaining_quantity -= allocated
        lot.last_changed_at = lot.last_changed_at or lot.created_at
        allocations.append((lot, allocated))
        remaining -= allocated
    if remaining:
        raise RuntimeError("La simulation a tenté de vendre un stock indisponible.")
    return allocations


def simulate(
    products: Sequence[Product],
    profiles: dict[int, ProductProfile],
    config: Config,
    rng: random.Random,
) -> Simulation:
    simulation = Simulation()
    promotion_list = generate_promotions(products, profiles, config, rng)
    simulation.promotions.extend(promotion_list)
    promotions_by_product: dict[int, list[Promotion]] = defaultdict(list)
    for promotion in promotion_list:
        promotions_by_product[promotion.source_product_id].append(promotion)

    lots_by_product: dict[int, list[Lot]] = defaultdict(list)
    product_by_id = {product.id: product for product in products}
    lot_id = 1
    cart_id = 1

    def add_lot(product: Product, quantity: int, timestamp: datetime) -> Lot:
        nonlocal lot_id
        lot = Lot(
            local_id=lot_id,
            product_id=product.id,
            created_at=timestamp,
            quantity=quantity,
            remaining_quantity=quantity,
            purchase_price=product.purchase_price,
            sale_price=product.sale_price,
            wholesale_price=product.wholesale_price,
            last_changed_at=timestamp,
        )
        lot_id += 1
        lots_by_product[product.id].append(lot)
        simulation.lots.append(lot)
        simulation.movements.append(Movement(lot, quantity, 0, "INCOMING", timestamp))
        return lot

    def available(product_id: int) -> int:
        return sum(lot.remaining_quantity for lot in lots_by_product[product_id])

    def ensure_stock(product: Product, needed: int, timestamp: datetime) -> None:
        profile = profiles[product.id]
        threshold = max(needed, math.ceil(profile.base_demand * config.reorder_threshold_days))
        if available(product.id) >= threshold:
            return
        target_days = config.reorder_threshold_days + rng.randint(16, 31)
        incoming_quantity = max(
            needed,
            math.ceil(profile.base_demand * target_days * rng.uniform(0.88, 1.16)),
            8,
        )
        add_lot(product, incoming_quantity, timestamp)

    for product in products:
        profile = profiles[product.id]
        initial_quantity = max(
            10,
            math.ceil(profile.base_demand * (config.reorder_threshold_days + 24)),
        )
        add_lot(product, initial_quantity, at(config.start_date, 7, rng.randint(0, 45)))

    for day_index in range(config.days):
        current_day = config.start_date + timedelta(days=day_index)
        for product in products:
            profile = profiles[product.id]
            opening_time = at(current_day, 7, rng.randint(0, 50))
            ensure_stock(product, 0, opening_time)

            timestamp = random_business_time(rng, current_day)
            promotion = active_promotion(promotions_by_product, product.id, timestamp)
            quantity = calculate_expected_demand(
                profile,
                current_day,
                day_index,
                config.days,
                promotion,
                rng,
            )
            simulation.daily_sales[(current_day, product.id)] += quantity
            simulation.daily_demand[(current_day, product.id)] += quantity

            for transaction_quantity in split_transactions(quantity, rng):
                sale_time = random_business_time(rng, current_day)
                transaction_promotion = active_promotion(
                    promotions_by_product,
                    product.id,
                    sale_time,
                )
                ensure_stock(product, transaction_quantity, sale_time - timedelta(minutes=5))
                lines = [
                    CartLine(lot, paid, 0, transaction_promotion)
                    for lot, paid in allocate_stock(lots_by_product[product.id], transaction_quantity)
                ]
                total_price = sum(
                    (line.lot.sale_price * line.paid_quantity for line in lines),
                    Decimal("0"),
                )
                if transaction_promotion:
                    gift_quantity = (
                        transaction_quantity // transaction_promotion.buy_quantity
                    ) * transaction_promotion.free_quantity
                    if gift_quantity:
                        gift_product = product_by_id[transaction_promotion.gift_product_id]
                        ensure_stock(gift_product, gift_quantity, sale_time - timedelta(minutes=3))
                        for lot, gifted in allocate_stock(
                            lots_by_product[gift_product.id],
                            gift_quantity,
                        ):
                            lines.append(CartLine(lot, 0, gifted, transaction_promotion))
                            simulation.daily_demand[(current_day, gift_product.id)] += gifted

                cart = CartPlan(
                    local_id=cart_id,
                    created_at=sale_time,
                    status="PAID",
                    lines=lines,
                    total_price=total_price.quantize(Decimal("0.01")),
                    reason=config.marker,
                    paid_at=sale_time + timedelta(minutes=rng.randint(1, 20)),
                )
                cart_id += 1
                simulation.carts.append(cart)
                for line in lines:
                    movement_type = "PROMOTION_GIFT" if line.free_quantity else "SALE"
                    simulation.movements.append(
                        Movement(
                            line.lot,
                            0,
                            line.paid_quantity + line.free_quantity,
                            movement_type,
                            cart.paid_at or sale_time,
                            cart,
                        ),
                    )

            # The live application represents an adjustment against an existing
            # lot and updates both its total and its remaining quantity.
            if rng.random() < config.adjustment_probability:
                usable_lots = [lot for lot in lots_by_product[product.id] if lot.remaining_quantity > 0]
                if usable_lots:
                    lot = rng.choice(usable_lots)
                    negative = rng.random() < 0.55
                    amount = min(lot.remaining_quantity, rng.randint(1, max(1, int(profile.base_demand)))) if negative else rng.randint(1, max(1, int(profile.base_demand)))
                    delta = -amount if negative else amount
                    lot.adjustment_delta += delta
                    lot.remaining_quantity += delta
                    adjustment_time = random_business_time(rng, current_day)
                    lot.last_changed_at = adjustment_time
                    simulation.movements.append(
                        Movement(
                            lot,
                            max(delta, 0),
                            max(-delta, 0),
                            "ADJUSTMENT",
                            adjustment_time,
                        ),
                    )

            # Cancellation is represented by Carts.status=CANCELLED in the
            # current application; it intentionally creates no stock movement.
            if rng.random() < config.cancellation_probability and lots_by_product[product.id]:
                lot = next((candidate for candidate in lots_by_product[product.id] if candidate.remaining_quantity), None)
                if lot:
                    cancelled_quantity = min(lot.remaining_quantity, rng.randint(1, 2))
                    simulation.carts.append(
                        CartPlan(
                            local_id=cart_id,
                            created_at=random_business_time(rng, current_day),
                            status="CANCELLED",
                            lines=[CartLine(lot, cancelled_quantity, 0, None)],
                            total_price=(lot.sale_price * cancelled_quantity).quantize(Decimal("0.01")),
                            reason=f"{config.marker} Annulation synthétique avant paiement.",
                            paid_at=None,
                        ),
                    )
                    cart_id += 1

    paid_lines = [
        (cart, line)
        for cart in simulation.carts
        if cart.status == "PAID"
        for line in cart.lines
        if line.paid_quantity > 0
    ]
    rng.shuffle(paid_lines)
    refunded_cart_ids: set[int] = set()
    for cart, line in paid_lines:
        if cart.local_id in refunded_cart_ids or rng.random() >= config.refund_probability:
            continue
        refund_day = cart.created_at.date() + timedelta(days=rng.randint(1, 7))
        if refund_day > config.end_date:
            continue
        refund_quantity = rng.randint(1, line.paid_quantity)
        simulation.refunds.append(
            Refund(line, cart, refund_quantity, random_business_time(rng, refund_day)),
        )
        refunded_cart_ids.add(cart.local_id)
    return simulation


def print_summary(
    simulation: Simulation,
    products: Sequence[Product],
    profiles: dict[int, ProductProfile],
    config: Config,
    product_load_report: ProductLoadReport,
) -> None:
    movement_counts = Counter(movement.movement_type for movement in simulation.movements)
    days = [config.start_date + timedelta(days=index) for index in range(config.days)]
    all_demands = [
        simulation.daily_demand[(day, product.id)]
        for day in days
        for product in products
    ]
    zero_sale_days = sum(
        simulation.daily_sales[(day, product.id)] == 0
        for day in days
        for product in products
    )
    promotion_days_by_product: dict[int, set[date]] = defaultdict(set)
    for promotion in simulation.promotions:
        promotion_day = promotion.start_at.date()
        while promotion_day <= promotion.end_at.date():
            promotion_days_by_product[promotion.source_product_id].add(promotion_day)
            promotion_days_by_product[promotion.gift_product_id].add(promotion_day)
            promotion_day += timedelta(days=1)
    cancelled = sum(cart.status == "CANCELLED" for cart in simulation.carts)
    print("\nPrévisualisation / statistiques")
    print(f"  Produits trouvés : {product_load_report.total_found}")
    print(f"  Produits retenus : {len(products)}")
    if product_load_report.excluded_products:
        print("  Produits exclus :")
        for excluded in product_load_report.excluded_products:
            print(
                f"    - {excluded.reference} (ID {excluded.id}, {excluded.name}) : "
                f"{excluded.reason}",
            )
    else:
        print("  Produits exclus : aucun")
    print(f"  Période simulée : {config.start_date} au {config.end_date} ({config.days} jours)")
    print(f"  Profil(s) : {dict(Counter(profile.name for profile in profiles.values()))}")
    for kind in ("SALE", "PROMOTION_GIFT", "INCOMING", "REFUND", "ADJUSTMENT"):
        count = movement_counts[kind] if kind != "REFUND" else len(simulation.refunds)
        print(f"  {kind} : {count}")
    print(f"  CANCELLATION : {cancelled} panier(s), 0 mouvement (comportement applicatif)")
    print(f"  Quantité totale vendue (SALE) : {sum(simulation.daily_sales.values())}")
    print(f"  Demande journalière min/max : {min(all_demands)} / {max(all_demands)}")
    print(f"  Jours produit sans vente : {zero_sale_days}")
    print("\nStatistiques détaillées par produit (demande = SALE + PROMOTION_GIFT) :")
    for product in products:
        daily_demands = [simulation.daily_demand[(day, product.id)] for day in days]
        total_demand = sum(daily_demands)
        zero_demand_days = sum(value == 0 for value in daily_demands)
        sale_quantity = sum(simulation.daily_sales[(day, product.id)] for day in days)
        promotion_gift_quantity = sum(
            movement.outgoing_quantity
            for movement in simulation.movements
            if movement.movement_type == "PROMOTION_GIFT"
            and movement.lot.product_id == product.id
        )
        incoming_movements = [
            movement
            for movement in simulation.movements
            if movement.movement_type == "INCOMING" and movement.lot.product_id == product.id
        ]
        print(
            f"  - {product.reference} (ID {product.id}) | {profiles[product.id].name}\n"
            f"      demande moy./min./max./écart-type : {statistics.fmean(daily_demands):.2f} / "
            f"{min(daily_demands)} / {max(daily_demands)} / "
            f"{statistics.pstdev(daily_demands):.2f}\n"
            f"      jours à zéro : {zero_demand_days} ({zero_demand_days / config.days * 100:.1f} %) | "
            f"SALE : {sale_quantity} | PROMOTION_GIFT : {promotion_gift_quantity}\n"
            f"      INCOMING : {len(incoming_movements)} mouvement(s), "
            f"{sum(movement.incoming_quantity for movement in incoming_movements)} unité(s)",
        )
        promotion_days = promotion_days_by_product[product.id]
        if promotion_days:
            promotion_demands = [
                simulation.daily_demand[(day, product.id)]
                for day in days
                if day in promotion_days
            ]
            non_promotion_demands = [
                simulation.daily_demand[(day, product.id)]
                for day in days
                if day not in promotion_days
            ]
            mean_during = statistics.fmean(promotion_demands)
            mean_outside = statistics.fmean(non_promotion_demands)
            uplift = (
                ((mean_during - mean_outside) / mean_outside) * 100
                if mean_outside > 0
                else None
            )
            uplift_label = f"{uplift:+.1f} %" if uplift is not None else "n/a (base hors promotion nulle)"
            print(
                f"      promotion ({len(promotion_days)} jour(s)) : moyenne hors/promotion "
                f"{mean_outside:.2f} / {mean_during:.2f} | uplift : {uplift_label}",
            )
    print("  Échantillon de mouvements :")
    for movement in sorted(simulation.movements, key=lambda item: item.created_at)[:12]:
        print(
            "    "
            f"{movement.created_at:%Y-%m-%d %H:%M} | lot {movement.lot.local_id} | "
            f"{movement.movement_type} | +{movement.incoming_quantity} / -{movement.outgoing_quantity}",
        )


def insert_many_returning_ids(cursor: psycopg.Cursor, query: str, rows: Sequence[tuple]) -> list[int]:
    if not rows:
        return []
    cursor.executemany(query, rows, returning=True)
    identifiers: list[int] = []
    for _ in cursor.results():
        returned = cursor.fetchone()
        if returned:
            identifiers.append(int(returned[0]))
    if len(identifiers) != len(rows):
        raise RuntimeError("PostgreSQL n'a pas retourné tous les identifiants insérés.")
    return identifiers


def persist(
    connection: psycopg.Connection,
    simulation: Simulation,
    config: Config,
    actor_id: int,
    supplier_id: int,
) -> None:
    """Persist all plans in one transaction; any exception rolls back everything."""
    with connection.transaction():
        with connection.cursor() as cursor:
            lot_rows = [
                (
                    lot.product_id,
                    lot.quantity + lot.adjustment_delta,
                    lot.created_at,
                    lot.last_changed_at or lot.created_at,
                    actor_id,
                    lot.purchase_price,
                    lot.sale_price,
                    actor_id,
                    supplier_id,
                    lot.wholesale_price,
                    lot.remaining_quantity,
                )
                for lot in simulation.lots
            ]
            lot_ids = insert_many_returning_ids(
                cursor,
                '''INSERT INTO "Inventories"
                   ("productId", quantity, "createdAt", "updatedAt", "createdBy", "purchasePrice",
                    "salePrice", "updatedBy", "supplierId", "wholeSalePrice", "remainingQuantity")
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                   RETURNING "ID"''',
                lot_rows,
            )
            for lot, database_id in zip(simulation.lots, lot_ids, strict=True):
                lot.database_id = database_id

            promotion_rows = [
                (
                    f"Synthetic demand {config.batch_key} #{promotion.local_id}",
                    promotion.start_at,
                    actor_id,
                    promotion.end_at,
                    promotion.buy_quantity,
                    promotion.free_quantity,
                    promotion.gift_product_id,
                )
                for promotion in simulation.promotions
            ]
            promotion_ids = insert_many_returning_ids(
                cursor,
                '''INSERT INTO "SpecialOffers"
                   (label, "createdAt", "createdBy", "updatedAt", "startDateTime", "endDateTime",
                    "buyQuantity", "freeQuantity", type, "productIdOffer")
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, 'BUY_X_GET_N', %s)
                   RETURNING "ID"''',
                [
                    (label, start, created_by, start, start, end, buy, free, product_offer)
                    for label, start, created_by, end, buy, free, product_offer in promotion_rows
                ],
            )
            promotion_database_ids = {
                promotion.local_id: database_id
                for promotion, database_id in zip(simulation.promotions, promotion_ids, strict=True)
            }
            association_rows = [
                (lot.database_id, promotion_database_ids[promotion.local_id], promotion.end_at)
                for promotion in simulation.promotions
                for lot in simulation.lots
                if lot.product_id == promotion.source_product_id and lot.created_at <= promotion.end_at
            ]
            if association_rows:
                cursor.executemany(
                    '''INSERT INTO "InventorySpecialOffers" ("inventoryId", "specialOfferId", "limitDate")
                       VALUES (%s, %s, %s)''',
                    association_rows,
                )

            cart_rows = [
                (
                    actor_id,
                    cart.created_at,
                    cart.paid_at or cart.created_at,
                    cart.status,
                    actor_id if cart.status == "PAID" else None,
                    cart.created_at if cart.status == "PAID" else None,
                    cart.paid_at,
                    cart.total_price,
                    "Client synthétique",
                    "CASH" if cart.status == "PAID" else None,
                    cart.reason,
                )
                for cart in simulation.carts
            ]
            cart_ids = insert_many_returning_ids(
                cursor,
                '''INSERT INTO "Carts"
                   ("soldBy", "createdAt", "updatedAt", status, "validatedBy", "validatedAt", "paidAt",
                    "totalPrice", "customerName", "paymentMethod", reason)
                   VALUES (%s, %s, %s, %s::"CartStatus", %s, %s, %s, %s, %s, %s::"PaymentMethod", %s)
                   RETURNING "ID"''',
                cart_rows,
            )
            for cart, database_id in zip(simulation.carts, cart_ids, strict=True):
                cart.database_id = database_id

            detail_rows: list[tuple] = []
            line_references: list[CartLine] = []
            for cart in simulation.carts:
                for line in cart.lines:
                    detail_rows.append(
                        (
                            cart.database_id,
                            line.lot.database_id,
                            line.paid_quantity,
                            line.free_quantity or None,
                            line.lot.sale_price,
                            Decimal("0") if line.free_quantity else line.lot.sale_price,
                            None,
                            False,
                            promotion_database_ids.get(line.promotion.local_id) if line.promotion else None,
                        ),
                    )
                    line_references.append(line)
            detail_ids = insert_many_returning_ids(
                cursor,
                '''INSERT INTO "CartDetails"
                   ("cartId", "inventoryId", quantity, "freeQuantity", "baseUnitPrice", "finalUnitPrice",
                    "discountAmount", "wholeSale", "specialOfferId")
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                   RETURNING "ID"''',
                detail_rows,
            )
            for line, database_id in zip(line_references, detail_ids, strict=True):
                line.database_id = database_id

            movement_rows = [
                (
                    movement.lot.database_id,
                    movement.incoming_quantity,
                    movement.outgoing_quantity,
                    actor_id,
                    movement.created_at,
                    movement.created_at,
                    movement.lot.purchase_price,
                    movement.lot.sale_price,
                    movement.movement_type,
                    movement.lot.wholesale_price,
                    movement.cart.database_id if movement.cart else None,
                )
                for movement in simulation.movements
            ]
            cursor.executemany(
                '''INSERT INTO "InventoryMovement"
                   ("inventoryId", "incomingQuantity", "outgoingQuantity", "actorId", "createdAt", "updatedAt",
                    "purchasePrice", "salePrice", type, "wholeSalePrice", "cartId")
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s::"InventoryMovementType", %s, %s)''',
                movement_rows,
            )

            for refund in simulation.refunds:
                cursor.execute(
                    '''UPDATE "CartDetails"
                       SET "refundedQuantity" = %s, "refundAt" = %s, "refundBy" = %s,
                           reason = 'Remboursement synthétique (produit non réintégré au stock)'
                       WHERE "ID" = %s''',
                    (refund.quantity, refund.created_at, actor_id, refund.line.database_id),
                )
                updated_total = (
                    refund.cart.total_price - refund.line.lot.sale_price * refund.quantity
                ).quantize(Decimal("0.01"))
                cursor.execute(
                    '''UPDATE "Carts"
                       SET status = 'PARTIALLY_REFUNDED', "totalPrice" = %s,
                           "updatedAt" = %s
                       WHERE "ID" = %s''',
                    (max(updated_total, Decimal("0")), refund.created_at, refund.cart.database_id),
                )
                cursor.execute(
                    '''INSERT INTO "InventoryMovement"
                       ("inventoryId", "incomingQuantity", "outgoingQuantity", "actorId", "createdAt", "updatedAt",
                        "purchasePrice", "salePrice", type, "wholeSalePrice", "cartId")
                       VALUES (%s, 0, 0, %s, %s, %s, %s, %s, 'REFUND', %s, %s)''',
                    (
                        refund.line.lot.database_id,
                        actor_id,
                        refund.created_at,
                        refund.created_at,
                        refund.line.lot.purchase_price,
                        refund.line.lot.sale_price,
                        refund.line.lot.wholesale_price,
                        refund.cart.database_id,
                    ),
                )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Simule sans modifier PostgreSQL.")
    parser.add_argument("--start-date", help="Remplace SYNTHETIC_START_DATE (YYYY-MM-DD).")
    parser.add_argument("--end-date", help="Remplace SYNTHETIC_END_DATE (YYYY-MM-DD).")
    parser.add_argument("--seed", type=int, help="Remplace SYNTHETIC_RANDOM_SEED.")
    return parser.parse_args()


def main() -> int:
    # The application keeps its DATABASE_URL in backend/.env. Environment values
    # already supplied by the shell always take precedence.
    load_dotenv(ROOT / "backend" / ".env", override=False)
    load_dotenv(ROOT / ".env", override=False)
    arguments = parse_arguments()
    try:
        ensure_not_production()
        config = load_config(arguments)
        database_url = os.getenv("DATABASE_URL")
        if not database_url:
            raise RuntimeError("DATABASE_URL est obligatoire.")
        connection_url, connection_options = postgres_connection_settings(database_url)
        with psycopg.connect(connection_url, options=connection_options) as connection:
            product_load_report = load_existing_products(connection, config)
            products = product_load_report.products
            if not products:
                print(
                    f"Produits trouvés : {product_load_report.total_found}; "
                    "produits retenus : 0. Aucune donnée n'a été créée.",
                )
                for excluded in product_load_report.excluded_products:
                    print(
                        f"  - {excluded.reference} (ID {excluded.id}) : {excluded.reason}",
                    )
                return 0
            existing = existing_batch_count(connection, config)
            if existing and not arguments.dry_run:
                raise RuntimeError(
                    f"Une génération synthétique pour cette période existe déjà ({existing} panier(s)). "
                    "Aucune écriture n'a été effectuée.",
                )
            actor_id, supplier_id = load_actor_and_supplier(connection, config)
            rng = random.Random(config.random_seed)
            profiles = build_product_profiles(products, rng)
            simulation = simulate(products, profiles, config, rng)
            print_summary(simulation, products, profiles, config, product_load_report)
            if arguments.dry_run:
                suffix = " (une génération existe déjà pour cette période)" if existing else ""
                print(f"\nDry-run terminé : PostgreSQL n'a pas été modifié.{suffix}")
                return 0
            persist(connection, simulation, config, actor_id, supplier_id)
            print(f"\nGénération terminée dans une transaction PostgreSQL. Marqueur : {config.marker}")
            return 0
    except (RuntimeError, ValueError, psycopg.Error) as error:
        print(f"Erreur : {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
