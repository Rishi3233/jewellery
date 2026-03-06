from dataclasses import dataclass
from typing import List
from PIL import Image

from .search import search_similar


@dataclass
class Concept:
    title: str
    description: str
    reference_product_id: str
    reference_image_path: str


def extract_style_tags(image: Image.Image) -> List[str]:
    # Placeholder: in a real setup, run a classifier to detect style attributes.
    return ["Floral", "Halo", "Antique Finish"]


def generate_concepts_from_image(image: Image.Image) -> List[Concept]:
    # Simple heuristic: take a few similar products and treat them as "concepts"
    similar = search_similar(image, top_k=4)
    concepts: List[Concept] = []
    tags = extract_style_tags(image)
    for i, r in enumerate(similar):
        concepts.append(
            Concept(
                title=f"Concept {i+1}: {', '.join(tags)}",
                description=(
                    f"Variation inspired by product {r.product_id} "
                    f"with {', '.join(tags).lower()}."
                ),
                reference_product_id=r.product_id,
                reference_image_path=r.image_path,
            )
        )
    return concepts
