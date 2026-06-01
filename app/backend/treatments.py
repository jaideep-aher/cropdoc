"""
treatments.py
-------------
Treatment recommendation database for PlantVillage disease classes.
Each entry maps a disease class name to structured treatment guidance.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Treatment:
    """Structured treatment recommendation for a plant disease."""
    disease_label: str
    severity_hint: str          # 'mild' | 'moderate' | 'severe'
    chemical_controls: list[str] = field(default_factory=list)
    biological_controls: list[str] = field(default_factory=list)
    cultural_controls: list[str] = field(default_factory=list)
    prevention: list[str] = field(default_factory=list)
    note: Optional[str] = None


# Keyed by PlantVillage class label (e.g. "Tomato___Early_blight")
TREATMENT_DB: dict[str, Treatment] = {
    "Apple___Apple_scab": Treatment(
        "Apple___Apple_scab", "moderate",
        chemical_controls=["Captan 50WP (apply at green tip)", "Myclobutanil (Rally)"],
        biological_controls=["Bacillus subtilis sprays"],
        cultural_controls=["Remove and destroy fallen leaves", "Improve canopy airflow via pruning"],
        prevention=["Plant scab-resistant varieties (Liberty, Enterprise)", "Avoid overhead irrigation"],
    ),
    "Apple___Black_rot": Treatment(
        "Apple___Black_rot", "severe",
        chemical_controls=["Captan", "Thiophanate-methyl"],
        cultural_controls=["Prune out mummified fruit and cankers", "Sanitize pruning tools"],
        prevention=["Remove dead wood promptly", "Maintain tree vigor with balanced fertilization"],
    ),
    "Apple___Cedar_apple_rust": Treatment(
        "Apple___Cedar_apple_rust", "moderate",
        chemical_controls=["Myclobutanil", "Propiconazole at pink bud stage"],
        cultural_controls=["Remove nearby eastern red cedar trees if feasible"],
        prevention=["Plant resistant apple varieties", "Apply protectant fungicides before infection periods"],
    ),
    "Apple___healthy": Treatment("Apple___healthy", "mild", note="No disease detected. Maintain routine care."),
    "Blueberry___healthy": Treatment("Blueberry___healthy", "mild", note="No disease detected."),
    "Cherry_(including_sour)___Powdery_mildew": Treatment(
        "Cherry_(including_sour)___Powdery_mildew", "moderate",
        chemical_controls=["Sulfur-based fungicides", "Potassium bicarbonate"],
        biological_controls=["Ampelomyces quisqualis (AQ10)"],
        cultural_controls=["Increase plant spacing", "Prune for airflow"],
        prevention=["Avoid excess nitrogen fertilization", "Use resistant rootstocks"],
    ),
    "Cherry_(including_sour)___healthy": Treatment("Cherry_(including_sour)___healthy", "mild"),
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": Treatment(
        "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot", "severe",
        chemical_controls=["Azoxystrobin + propiconazole (Quilt Xcel)", "Prothioconazole"],
        cultural_controls=["Rotate with non-host crops (soybeans)", "Till under infected residue"],
        prevention=["Plant tolerant hybrids", "Avoid continuous corn"],
    ),
    "Corn_(maize)___Common_rust_": Treatment(
        "Corn_(maize)___Common_rust_", "moderate",
        chemical_controls=["Triazole fungicides at early tassel"],
        cultural_controls=["Scout fields weekly during silking"],
        prevention=["Plant resistant hybrids", "Early planting to avoid peak spore periods"],
    ),
    "Corn_(maize)___Northern_Leaf_Blight": Treatment(
        "Corn_(maize)___Northern_Leaf_Blight", "severe",
        chemical_controls=["Propiconazole", "Azoxystrobin"],
        cultural_controls=["Crop rotation", "Bury infected residue"],
        prevention=["Resistant hybrids (Ht gene)", "Avoid dense plant populations"],
    ),
    "Corn_(maize)___healthy": Treatment("Corn_(maize)___healthy", "mild"),
    "Grape___Black_rot": Treatment(
        "Grape___Black_rot", "severe",
        chemical_controls=["Mancozeb", "Myclobutanil at 1-inch shoot growth"],
        cultural_controls=["Remove mummified berries", "Improve trellis for airflow"],
        prevention=["Apply fungicides at 7-10 day intervals during wet weather"],
    ),
    "Grape___Esca_(Black_Measles)": Treatment(
        "Grape___Esca_(Black_Measles)", "severe",
        chemical_controls=["Sodium arsenite (restricted use)"],
        cultural_controls=["Prune in dry weather", "Paint wounds with wound sealant"],
        prevention=["Use certified disease-free planting material", "Avoid water stress"],
        note="No fully effective chemical cure; management focuses on prevention.",
    ),
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": Treatment(
        "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)", "moderate",
        chemical_controls=["Copper-based fungicides", "Mancozeb"],
        cultural_controls=["Remove infected leaves", "Improve vineyard drainage"],
    ),
    "Grape___healthy": Treatment("Grape___healthy", "mild"),
    "Orange___Haunglongbing_(Citrus_greening)": Treatment(
        "Orange___Haunglongbing_(Citrus_greening)", "severe",
        chemical_controls=["Insecticides targeting Asian citrus psyllid vector (imidacloprid)"],
        cultural_controls=["Remove and destroy infected trees immediately"],
        prevention=["Use certified disease-free budwood", "Monitor and control psyllid populations"],
        note="No cure exists; eradication of infected trees is the only management option.",
    ),
    "Peach___Bacterial_spot": Treatment(
        "Peach___Bacterial_spot", "moderate",
        chemical_controls=["Copper hydroxide (early season)", "Oxytetracycline"],
        cultural_controls=["Prune for airflow", "Avoid overhead irrigation"],
        prevention=["Plant resistant varieties", "Apply copper sprays at dormancy"],
    ),
    "Peach___healthy": Treatment("Peach___healthy", "mild"),
    "Pepper,_bell___Bacterial_spot": Treatment(
        "Pepper,_bell___Bacterial_spot", "moderate",
        chemical_controls=["Copper bactericides + mancozeb tank mix"],
        cultural_controls=["Remove infected plant debris", "Avoid working when plants are wet"],
        prevention=["Use disease-free transplants", "Rotate crops 2-3 years"],
    ),
    "Pepper,_bell___healthy": Treatment("Pepper,_bell___healthy", "mild"),
    "Potato___Early_blight": Treatment(
        "Potato___Early_blight", "moderate",
        chemical_controls=["Chlorothalonil", "Azoxystrobin"],
        cultural_controls=["Destroy volunteer potatoes", "Hill soil around plants"],
        prevention=["Use certified seed potatoes", "Adequate fertilization to reduce stress"],
    ),
    "Potato___Late_blight": Treatment(
        "Potato___Late_blight", "severe",
        chemical_controls=["Metalaxyl + chlorothalonil", "Cymoxanil (Curzate)"],
        cultural_controls=["Destroy infected plant material immediately", "Avoid excessive irrigation"],
        prevention=["Plant resistant varieties", "Apply protectant fungicides prophylactically"],
        note="Late blight spreads rapidly. Act within 24 hours of detection.",
    ),
    "Potato___healthy": Treatment("Potato___healthy", "mild"),
    "Raspberry___healthy": Treatment("Raspberry___healthy", "mild"),
    "Soybean___healthy": Treatment("Soybean___healthy", "mild"),
    "Squash___Powdery_mildew": Treatment(
        "Squash___Powdery_mildew", "moderate",
        chemical_controls=["Potassium bicarbonate", "Sulfur fungicides"],
        biological_controls=["Bacillus subtilis (Serenade)"],
        cultural_controls=["Increase plant spacing", "Water at base of plant"],
        prevention=["Plant resistant varieties", "Avoid nitrogen excess"],
    ),
    "Strawberry___Leaf_scorch": Treatment(
        "Strawberry___Leaf_scorch", "moderate",
        chemical_controls=["Captan", "Thiram"],
        cultural_controls=["Remove old leaves after harvest", "Renovate bed with mowing"],
        prevention=["Plant certified disease-free runners", "Avoid overhead irrigation"],
    ),
    "Strawberry___healthy": Treatment("Strawberry___healthy", "mild"),
    "Tomato___Bacterial_spot": Treatment(
        "Tomato___Bacterial_spot", "moderate",
        chemical_controls=["Copper bactericide + mancozeb", "Acibenzolar-S-methyl (Actigard)"],
        cultural_controls=["Stake and trellis to improve airflow", "Sanitize tools"],
        prevention=["Use disease-free transplants", "Avoid handling wet plants"],
    ),
    "Tomato___Early_blight": Treatment(
        "Tomato___Early_blight", "moderate",
        chemical_controls=["Chlorothalonil", "Azoxystrobin (Quadris)"],
        cultural_controls=["Stake plants", "Mulch to prevent soil splash"],
        prevention=["Rotate crops 2 years", "Remove infected lower leaves"],
    ),
    "Tomato___Late_blight": Treatment(
        "Tomato___Late_blight", "severe",
        chemical_controls=["Metalaxyl (Ridomil)", "Cymoxanil", "Famoxadone + cymoxanil (Tanos)"],
        cultural_controls=["Destroy infected tissue immediately", "Avoid overhead watering"],
        prevention=["Plant resistant varieties (Mountain Magic, Defiant)", "Scout regularly"],
        note="Highly contagious. Notify neighboring farms if detected.",
    ),
    "Tomato___Leaf_Mold": Treatment(
        "Tomato___Leaf_Mold", "moderate",
        chemical_controls=["Chlorothalonil", "Copper-based fungicide"],
        cultural_controls=["Reduce greenhouse humidity below 85%", "Prune lower leaves"],
        prevention=["Resistant varieties", "Ensure adequate ventilation"],
    ),
    "Tomato___Septoria_leaf_spot": Treatment(
        "Tomato___Septoria_leaf_spot", "moderate",
        chemical_controls=["Chlorothalonil", "Mancozeb"],
        cultural_controls=["Remove infected leaves", "Avoid overhead watering"],
        prevention=["Crop rotation", "Mulch around base"],
    ),
    "Tomato___Spider_mites Two-spotted_spider_mite": Treatment(
        "Tomato___Spider_mites Two-spotted_spider_mite", "moderate",
        chemical_controls=["Abamectin (Agri-Mek)", "Bifenazate (Acramite)"],
        biological_controls=["Phytoseiulus persimilis predatory mite", "Neoseiulus californicus"],
        cultural_controls=["Increase humidity", "Spray plants with strong water stream"],
        prevention=["Avoid water stress", "Monitor with sticky traps"],
    ),
    "Tomato___Target_Spot": Treatment(
        "Tomato___Target_Spot", "moderate",
        chemical_controls=["Boscalid + pyraclostrobin (Pristine)", "Azoxystrobin"],
        cultural_controls=["Stake and trellis plants", "Avoid leaf wetness"],
    ),
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": Treatment(
        "Tomato___Tomato_Yellow_Leaf_Curl_Virus", "severe",
        chemical_controls=["Insecticides targeting whitefly vector (imidacloprid, thiamethoxam)"],
        cultural_controls=["Remove and destroy infected plants", "Use yellow sticky traps"],
        prevention=["Use virus-resistant varieties", "Reflective mulch to repel whiteflies"],
        note="No cure for infected plants. Vector control is key.",
    ),
    "Tomato___Tomato_mosaic_virus": Treatment(
        "Tomato___Tomato_mosaic_virus", "severe",
        cultural_controls=["Remove infected plants", "Disinfect tools with bleach solution"],
        prevention=["Use TMV-resistant seeds", "Wash hands before handling plants"],
        note="No chemical cure. Prevention is the only management strategy.",
    ),
    "Tomato___healthy": Treatment("Tomato___healthy", "mild"),
}


def get_treatment(class_name: str) -> Treatment:
    """
    Look up treatment for a given PlantVillage class label.

    Args:
        class_name: Disease class string (e.g. 'Tomato___Early_blight').

    Returns:
        Treatment dataclass. Falls back to a generic healthy treatment if not found.
    """
    return TREATMENT_DB.get(
        class_name,
        Treatment(class_name, "mild", note="No specific treatment data available. Consult an agronomist."),
    )
