from pydantic import BaseModel
from typing import Literal


class PredictionInput(BaseModel):
    generation: int
    log_height_m: float
    log_weight_kg: float

    evo_stage: int
    is_mega: bool
    is_primal: bool
    is_form: bool

    status: Literal["mythical", "normal", "sub_legendary"]

    type_1: Literal[
        "dark","dragon","electric","fairy","fighting","fire",
        "flying","ghost","grass","ground","ice","normal",
        "poison","psychic","rock","steel","water"
    ]

    type_2: Literal[
        "dark","dragon","electric","fairy","fighting","fire",
        "flying","ghost","grass","ground","ice","normal",
        "poison","psychic","rock","steel","water","none"
    ] | None = None

    growth_rate: Literal[
        "fast",
        "medium_fast",
        "medium_slow",
        "slow",
        "fluctuating"
    ]
