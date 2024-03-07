from PIL import Image
from ultralytics import YOLO

# Load a pretrained YOLOv8n model
model = YOLO('yolov8n.pt')

# Open an image using PIL
source = Image.open('../task4/trickyimg/000000062808.jpg')

# Run inference on the source
results = model(source)
for i, r in enumerate(results):
    # Plot results image
    im_bgr = r.plot()  # BGR-order numpy array
    im_rgb = Image.fromarray(im_bgr[..., ::-1])  # RGB-order PIL image

    # Show results to screen (in supported environments)
    r.show()

    # Save results to disk
    # r.save(filename=f'results{i}.jpg')