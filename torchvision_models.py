from torchvision.io.image import read_image
from torchvision.models.detection import ssdlite320_mobilenet_v3_large,SSDLite320_MobileNet_V3_Large_Weights
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import to_pil_image
import torch
import torchvision

img = read_image("../task4/trickyimg/000000062808.jpg")

# Step 1: Initialize model with the best available weights
weights = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
model = ssdlite320_mobilenet_v3_large(weights=weights)
model.eval()

# Step 2: Initialize the inference transforms
preprocess = weights.transforms()

# Step 3: Apply inference preprocessing transforms
batch = [preprocess(img)]

# Step 4: Use the model and visualize the prediction
with torch.no_grad():
    prediction = model(batch)[0]

keep = prediction["scores"] >= 0.2
prediction["boxes"] = prediction["boxes"][keep]
prediction["scores"] = prediction["scores"][keep]
prediction["labels"] = prediction["labels"][keep]

boxes = prediction["boxes"]
# scores = prediction["scores"]
labels = [weights.meta["categories"][i] for i in prediction["labels"]]
#
# keep = torchvision.ops.nms(boxes, scores, iou_threshold=0.9)

box = draw_bounding_boxes(img, boxes=boxes,
                          labels=labels,
                          colors="red",
                          width=4, font_size=30)
print(prediction["boxes"].shape)
im = to_pil_image(box.detach())
im.show()