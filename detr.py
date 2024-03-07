from transformers import DetrImageProcessor, DetrForObjectDetection
import torch
from PIL import Image,ImageDraw
import os

#Images are resized/rescaled such that the shortest side is at least 800 pixels and the largest side at most 1333 pixels,
# and normalized across the RGB channels with the ImageNet mean (0.485, 0.456, 0.406) and standard deviation (0.229, 0.224, 0.225)

def detr_inf(file_pth,i):
    # file_pth = 'object_detection_imgs/02.jpg'
    image = Image.open(file_pth)

    # you can specify the revision tag if you don't want the timm dependency
    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
    model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", revision="no_timm")

    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)

    # convert outputs (bounding boxes and class logits) to COCO API
    # let's only keep detections with score > 0.9
    target_sizes = torch.tensor([image.size[::-1]])
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]

    draw = ImageDraw.Draw(image)
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box = [round(i, 2) for i in box.tolist()]
        draw.rectangle(box, outline="red", width=2)
        label_text = f"{model.config.id2label[label.item()]}: {round(score.item(), 3)}"

        draw.text((box[0], box[1]), label_text, fill="red")
        # print(
        #         f"Detected {model.config.id2label[label.item()]} with confidence "
        #         f"{round(score.item(), 3)} at location {box}"
        # )
    # output_pth = 'object_detection_imgs'
    annotated_image_path = f'object_detection_imgs/detr_result{i}.jpg'
    image.save(annotated_image_path)

    print(f"Annotated image saved at: {annotated_image_path}")