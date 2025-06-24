from db.database import SessionLocal, init_db
from db.models import Image, Model, Experiment, LabelStatusEnum, ModelTypeEnum, ModelStatusEnum
import uuid

# Initialize DB and create tables
init_db()
db = SessionLocal()

# Create a sample image
image1 = Image(
    id=str(uuid.uuid4()),
    image_path="data_pipeline/input/sample_image.jpg",
    label_path="data_pipeline/labeled/sample_image.json",
    classes=["FireBSI", "SmokeBSI"],
    bboxes=[[100, 200, 300, 400], [120, 210, 310, 410]],
    label_status=LabelStatusEnum.auto,
    semi_supervised={"source": "YOLO", "confidence": [0.92, 0.85]},
    tags=["smoke"]
)

image2 = Image(
    id=str(uuid.uuid4()),
    image_path="data_pipeline/input/val_image.jpg",
    label_path="data_pipeline/labeled/val_image.json",
    classes=["VehicleBSI"],
    bboxes=[[50, 60, 200, 220]],
    label_status=LabelStatusEnum.human,
    semi_supervised={"source": "YOLO", "confidence": [0.65]},
    tags=["validation"]
)

image3 = Image(
    id=str(uuid.uuid4()),
    image_path="data_pipeline/input/test_image.jpg",
    label_path="data_pipeline/labeled/test_image.json",
    classes=["PersonBSI"],
    bboxes=[[20, 40, 180, 210]],
    label_status=LabelStatusEnum.auto,
    semi_supervised={"source": "DINO", "confidence": [0.75]},
    tags=["test"]
)

# Add images
db.add_all([image1, image2, image3])
db.commit()

# Create a model using the above images
model = Model(
    id=str(uuid.uuid4()),
    model_path="model_registry/model/model.pt",
    model_type=ModelTypeEnum.base,
    model_status=ModelStatusEnum.trained,
    train_image_ids=[image1.id],
    val_image_ids=[image2.id],
    test_image_ids=[image3.id],
    training_history={"epochs": 1, "loss": 0.02}
)
db.add(model)
db.commit()
db.refresh(model)

# Create an experiment using the model
experiment = Experiment(
    id=str(uuid.uuid4()),
    model_id=model.id,
    augmentation={"flip": True, "blur": False},
    random_seed="42",
    performance={"mAP": 0.89, "precision": 0.91},
    config={"batch_size": 16, "imgsz": 640}
)
db.add(experiment)
db.commit()

print("✅ Test data inserted successfully!")
db.close()
