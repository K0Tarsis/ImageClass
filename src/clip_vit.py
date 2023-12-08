import numpy as np
from transformers import CLIPProcessor, CLIPModel

model = CLIPModel.from_pretrained("../clip-vit-large-patch14")
processor = CLIPProcessor.from_pretrained("../clip-vit-large-patch14")


def predict_classes_clip(classes, img):
    inputs = processor(text=classes, images=img, return_tensors="pt", padding=True)

    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1)

    probe = probs[0].detach().numpy()

    to_return = {}

    for index, number in enumerate(probe):

        # if number > 0.3:
        to_return.update({classes[index].replace("a photo of a ", ""): str(number)})

    # indx = np.argmax(probe)
    # return classes[indx].replace("a photo of a ", "")

    return to_return




