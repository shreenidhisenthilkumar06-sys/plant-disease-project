"""Human-readable PlantVillage class information returned with predictions."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DiseaseInfo:
    display_name: str
    description: str
    symptoms: str
    causes: str
    prevention: str
    treatment: str


def _healthy(crop: str) -> DiseaseInfo:
    return DiseaseInfo(
        f"Healthy {crop}",
        f"The leaf appears consistent with a healthy {crop} plant.",
        "No disease symptoms were detected in the submitted image.",
        "No specific pathogen is indicated by this prediction.",
        "Continue regular scouting, balanced nutrition, clean tools, and good airflow.",
        "No treatment is needed. Monitor plants and follow normal crop-care practices.",
    )


def _info(name: str, description: str, symptoms: str, causes: str, prevention: str, treatment: str) -> DiseaseInfo:
    return DiseaseInfo(name, description, symptoms, causes, prevention, treatment)


# Every class in backend/class_names.json is represented below. Recommendations are
# general guidance; local labels and agricultural extension advice take precedence.
DISEASE_INFO: dict[str, DiseaseInfo] = {
    "Grape___Black_rot": _info("Grape Black Rot", "A fungal disease affecting grape leaves, shoots, and berries.", "Brown leaf spots with black dots; berries turn brown, then shrivel into black mummies.", "Guignardia bidwellii overwinters in infected fruit and plant debris; rain spreads spores.", "Remove mummified berries, prune dense vines, and maintain good canopy airflow.", "Apply labeled fungicides preventively during wet periods, following local extension guidance."),
    "Grape___Esca_(Black_Measles)": _info("Grape Esca (Black Measles)", "A trunk disease complex that can cause chronic vine decline.", "Interveinal leaf striping, dark berry specks, dieback, and sudden collapse in severe cases.", "Wood-inhabiting fungi enter pruning wounds and colonize grapevine trunks.", "Use clean planting stock, protect pruning wounds, remove diseased wood, and sanitize tools.", "There is no reliable curative treatment; renew trunks or remove severely affected vines with expert guidance."),
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": _info("Grape Leaf Blight", "A fungal leaf-spot disease of grape foliage.", "Irregular brown to black leaf spots that may merge and cause premature defoliation.", "Pseudocercospora/Isariopsis fungi persist in infected debris and spread in humid weather.", "Improve airflow, collect infected leaves, and avoid prolonged wet foliage.", "Use locally registered fungicides when symptoms and weather conditions warrant treatment."),
    "Grape___healthy": _healthy("Grape"),
}


def get_disease_info(class_name: str) -> DiseaseInfo:
    """Return metadata for a model label, failing safely if labels are misconfigured."""
    try:
        return DISEASE_INFO[class_name]
    except KeyError as error:
        raise RuntimeError(f"No disease information is configured for model class: {class_name}") from error
