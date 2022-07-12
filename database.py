import argparse
from re import L

import cv2
import os
import numpy as np
import torch
import json
from backbones import get_model

def evaluation_image(net, img: np.ndarray):
    img = cv2.imread(img)
    img = cv2.resize(img, (112, 112))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.transpose(img, (2, 0, 1))
    img = torch.from_numpy(img).unsqueeze(0).float()
    img.div_(255).sub_(0.5).div_(0.5)
    emb_vector = net(img).detach().numpy()
    return emb_vector

def inference(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = get_model(args.network, fp16=False)
    net.load_state_dict(torch.load(args.weight, map_location=device))
    net.to(device)
    net.eval()
    
    # Tạo file .json
    if os.path.exists('database.json'):
        os.remove('database.json')  

    # Database là kiểu dữ liệu dict : key là label, values là tập features
    database = {}
    if args.image != None:
        evaluation_image(net, args.image)
    elif args.images != None:
        for path in os.listdir(args.images):
            pth = os.path.join(args.images, path)
            if os.path.isdir(pth):
                emb_label = []
                for img in os.listdir(pth):
                    path_img = os.path.join(pth, img)
                    emb_label.append(evaluation_image(net, path_img).tolist())
                database[path] = emb_label
         
        # Writing to sample.json
        with open("database.json", "w") as outfile:
            json.dump(database, outfile)
    
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch ArcFace Evaluation')
    parser.add_argument('--network', type=str, default='r50', help='backbone network')
    parser.add_argument('--weight', type=str, default='')
    parser.add_argument('--image', type=str, default=None,
                        help='Run one image when read path file image')
    parser.add_argument('--images', type=str, default=None,
                        help='Run folder image when read folder image')
    parser.add_argument('--save', type=bool, default=True)
    args = parser.parse_args()
    inference(args)
