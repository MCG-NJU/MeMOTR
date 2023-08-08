# Copyright (c) Ruopeng Gao. All Rights Reserved.
import matplotlib.pyplot as plt

from torchvision import transforms


def visualize_a_batch(batch: dict):
    for i in range(len(batch["infos"])):
        for j in range(len(batch["infos"][i])):
            img = transforms.ToPILImage()(batch["infos"][i][j]["unnorm_img"])
            fig, ax = plt.subplots(1)
            for box in batch["infos"][i][j]["boxes"]:
                img_w, img_h = img.size
                w, h = box[2].item(), box[3].item()
                x, y = (box[0].item() - w / 2) * img_w, (box[1].item() - h / 2) * img_h
                rect = plt.Rectangle((x, y), w*img_w, h*img_h, fill=False, edgecolor="red", linewidth=1)
                ax.add_patch(rect)
            plt.imshow(img)
            plt.show()
    pass
