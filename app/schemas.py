from pydantic import BaseModel
from typing import Literal


class PredictionInput(BaseModel):
    generation: int
    height_m: float
    weight_kg: float

    evo_stage: int
    is_mega: bool
    is_primal: bool
    is_form: bool

    status: Literal["legendary", "mythical", "normal", "sub_legendary"]

    type_1: Literal[
        "bug","dark","dragon","electric","fairy","fighting","fire",
        "flying","ghost","grass","ground","ice","normal",
        "poison","psychic","rock","steel","water"
    ]

    type_2: Literal[
        "bug","dark","dragon","electric","fairy","fighting","fire",
        "flying","ghost","grass","ground","ice","normal",
        "poison","psychic","rock","steel","water","none"
    ] | None = None

    growth_rate: Literal[
        "erratic",
        "fast",
        "medium_fast",
        "medium_slow",
        "slow",
        "fluctuating"
    ]
