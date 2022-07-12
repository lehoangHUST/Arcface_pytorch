import argparse

import cv2
import numpy as np
import torch
import os
from backbones import get_model

def cosine_similarity(emb1, emb2):
    return np.sum((emb1 * emb2)) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))


def distance_similarity(emb1, emb2):
    return float(1/512) * np.sum((emb1-emb2)**2)**0.5


@torch.no_grad()
def inference(net, img):
    if img is None:
        img = np.random.randint(0, 255, size=(112, 112, 3), dtype=np.uint8)
    else:
        img = cv2.imread(img)
        img = cv2.resize(img, (112, 112))

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.transpose(img, (2, 0, 1))
    img = torch.from_numpy(img).unsqueeze(0).float()
    img.div_(255).sub_(0.5).div_(0.5)
    emb_vector = net(img).numpy()
    return emb_vector

@torch.no_grad()
def vertification(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = get_model(args.network, fp16=False)
    net.load_state_dict(torch.load(args.weight, map_location=device))
    net.to(device)
    net.eval()
    # Emb vector1
    emb_vector1 = inference(net, args.img1)
    # Emb for dir
    for path in os.listdir(args.images):
        emb_vector2 = inference(net, os.path.join(args.images, path))
    
        similarity = cosine_similarity(emb_vector1, emb_vector2)
        print(similarity)
        if similarity >= args.threshold:
            print(f"Two images is same person")
        else:
            print(f"Two images is difference person")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch ArcFace Evaluation')
    parser.add_argument('--network', type=str, default='r50', help='backbone network')
    parser.add_argument('--weight', type=str, default='')
    parser.add_argument('--threshold', type=float, default=0.1)
    parser.add_argument('--img1', type=str, default=None)
    parser.add_argument('--img2', type=str, default=None)
    parser.add_argument('--images', type=str, default=None)
    args = parser.parse_args()
    vertification(args)
