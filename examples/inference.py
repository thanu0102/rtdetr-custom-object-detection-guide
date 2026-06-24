from ultralytics import RTDETR

model = RTDETR("best.pt")

results = model.predict(
    source="test.jpg",
    save=True,
    conf=0.25
)

print("Inference completed.")