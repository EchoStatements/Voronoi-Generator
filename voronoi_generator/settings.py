"""Schema for config files."""
from typing import Optional, Union

from pydantic import BaseModel, Field


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

    file_path: str = Field("images/example.png")

    numpy_seed: int = Field(default=0)
    python_seed: int = Field(default=0)


class NamedColours(BaseModel):
    """Schema for colour list."""

    named_colours: dict[str, list[int]]
