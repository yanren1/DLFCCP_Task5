from PIL import Image
from ultralytics import YOLO

def yolov8_inf(file_pth,i):
    # Load a pretrained YOLOv8n model

    model = YOLO('yolov8n.pt')

    # Open an image using PIL
    # file_pth = 'object_detection_imgs/02.jpg'
    source = Image.open(file_pth)

    # Run inference on the source
    results = model(source)
    for _, r in enumerate(results):
        # Plot results image
        im_bgr = r.plot()  # BGR-order numpy array
        im_rgb = Image.fromarray(im_bgr[..., ::-1])  # RGB-order PIL image

        # Show results to screen (in supported environments)
        # r.show()

        # Save results to disk
        r.save(filename=f'object_detection_imgs/yolov8n_results{i}.jpg')