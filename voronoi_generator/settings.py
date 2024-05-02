"""Schema for config files."""
from typing import Literal, Optional, Union

from pydantic import BaseModel, Field

SamplingMethod = Literal["uniform", "poisson"]


class VoronoiDiagramSettings(BaseModel):
    """Schema for config file."""

    x_size: int = Field(default=500)
    y_size: int = Field(default=500)
    n_centroids: int = Field(default=30)

    distance_function: str = "cityblock"

    colour_list: list[Union[list[int], str]] = Field()
    named_colours_file: Optional[str] = Field(default=None)

    wrap_x: bool = Field(default=True)

    border_thickness: int = Field(default=2)

    file_path: Optional[str] = Field(default=None)

    numpy_seed: int = Field(default=0)
    python_seed: int = Field(default=0)

    sampling_method: SamplingMethod = Field(default="uniform")


class NamedColours(BaseModel):
    """Schema for colour list."""

    named_colours: dict[str, list[int]]
