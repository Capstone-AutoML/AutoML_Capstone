from sqlalchemy import Column, String, Enum, JSON, ForeignKey
from sqlalchemy.orm import declarative_base, relationship
import enum
from db.base import Base


# --- ENUMS ---

class LabelStatusEnum(enum.Enum):
    auto = "auto"
    human = "human"
    mismatch = "mismatch"

class ModelTypeEnum(enum.Enum):
    base = "base"
    distilled = "distilled"
    quantized = "quantized"

class ModelStatusEnum(enum.Enum):
    training = "training"
    trained = "trained"

# --- TABLES ---

class Image(Base):
    __tablename__ = "images"
    id = Column(String, primary_key=True)  # UUID string
    image_path = Column(String, nullable=False)  # Path to the real image
    label_path = Column(String)  # Path to the label JSON
    classes = Column(JSON)  # Array of class labels
    bboxes = Column(JSON)  # Array of bounding boxes [[x1, y1, x2, y2], ...]
    label_status = Column(Enum(LabelStatusEnum))
    semi_supervised = Column(JSON)  # JSON field for extra auto-label info
    tags = Column(JSON)  # Array of strings like ["augmented", "smoke"]

class Model(Base):
    __tablename__ = "models"
    id = Column(String, primary_key=True)  # UUID string
    model_path = Column(String, nullable=False)
    model_type = Column(Enum(ModelTypeEnum))  # base, distilled, quantized
    model_status = Column(Enum(ModelStatusEnum))  # training, trained
    train_image_ids = Column(JSON)  # List of image UUIDs used in training
    val_image_ids = Column(JSON)  # List of image UUIDs used in validation
    test_image_ids = Column(JSON)  # List of image UUIDs used in testing
    training_history = Column(JSON)  # JSON dict of training metrics, loss, etc.

class Experiment(Base):
    __tablename__ = "experiments"
    id = Column(String, primary_key=True)  # UUID string
    model_id = Column(String, ForeignKey("models.id"))
    augmentation = Column(JSON)  # JSON object (e.g. {"flip": True, "blur": False})
    random_seed = Column(String)
    performance = Column(JSON)  # e.g. {"mAP": 0.89, "precision": 0.91}
    config = Column(JSON)  # e.g. {"batch_size": 16, "imgsz": 640}
