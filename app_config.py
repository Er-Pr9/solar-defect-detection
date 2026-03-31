MODEL_PATH = "solar_defect_model_final.pth"
IMAGE_SIZE = 224

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

CLASS_INFO = {
    "black_core": "Dark inactive region in the solar cell.",
    "crack": "Fracture or crack-like defect in the solar cell.",
    "finger": "Defect affecting the thin metallization finger lines.",
    "horizontal_dislocation": "Horizontal structural or pattern displacement in the cell.",
    "short_circuit": "Region suggesting electrical shorting or abnormal connection.",
    "star_crack": "Star-shaped crack pattern radiating from a point.",
    "thick_line": "Abnormally thick line-type defect pattern.",
    "vertical_dislocation": "Vertical structural or pattern displacement in the cell.",
}
