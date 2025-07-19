import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

from PIL import Image
import requests

from transformers import CLIPProcessor, CLIPModel

model = CLIPModel.from_pretrained("clip-vit")
processor = CLIPProcessor.from_pretrained("clip-vit")

breakpoint()

img_path = "demo_images/cat.jpg"
image = Image.open(img_path)

inputs = processor(
    text = ["a photo of a cat", "a photo of a dog"], 
    images = image, 
    return_tensors = "pt", 
    padding = True
)

# add output
inputs['output_attentions'] = True
inputs['output_hidden_states'] = True

for keys in inputs.keys():
    print(keys)

breakpoint()

outputs = model(**inputs)
logits_per_image = outputs.logits_per_image # this is the image-text similarity score
probs = logits_per_image.softmax(dim=1) # we can take the softmax to get the label probabilities
print(probs)

vision_model_output = outputs.vision_model_output
attentions = vision_model_output.attentions

breakpoint()