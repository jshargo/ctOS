from ultralytics import YOLO


model = YOLO("yolov8m.pt")  


train_results = model.train(
    data="path/to/your/custom_dataset.yaml",  # Replace with your dataset YAML file
    epochs=100,         # Number of training epochs
    imgsz=640,          # Training image size
    device="cpu",       # Use 'cuda' if you have a GPU
)

# Evaluate model performance on the validation set
metrics = model.val()

# Save the trained model
model.save("best_model.pt")
