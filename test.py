import argparse
import cv2
import torch
import json
import os
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from backbones import get_model

# Angle between two vector : Cosine similarity
def cosine_similarity(emb1, emb2):
    return np.sum((emb1 * emb2)) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

# Euclid distance similarity
def distance_similarity(emb1, emb2):
    return np.sum((emb1-emb2)**2)**0.5


def inference(recognition, img, database):
    # Recognition object bbox in image
    img = cv2.resize(img, (112, 112))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.transpose(img, (2, 0, 1))
    img = torch.from_numpy(img).unsqueeze(0).float().to('cuda')
    img.div_(255).sub_(0.5).div_(0.5)
    emb_vector = recognition(img).detach().cpu().numpy()
    find_label = "Unknown"
    data = {}
    for label, emb_database in database.items():
        emb_database = np.array(emb_database)
        emb_database = emb_database.reshape(emb_database.shape[0], emb_database.shape[2])
        _max = -1.0
        for idx in emb_database:
            similarity = cosine_similarity(emb_vector, idx)
            if _max < similarity:
                _max = float(similarity)
                
        data[label] = _max


    maxkey_similarity = max(data, key=data.get)
    return maxkey_similarity

def evaluate(images, recognition, database):

    pred_label = []
    gt_label = []
    for gt,ids in enumerate(os.listdir(images)):
        path_ids = os.path.join(images, ids)
        for _id_ in os.listdir(path_ids):
            img = cv2.imread(os.path.join(path_ids, _id_))
            gt_label.append(gt+1)
            pred_label.append(inference(recognition, img, database))
    print('Accuracy:', accuracy_score(gt_label, pred_label))
    print('Recall:', recall_score(gt_label, pred_label,
                              average='weighted'))

    print('Precision:', precision_score(gt_label, pred_label,
                                    average='weighted'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch ArcFace Evaluation')
    parser.add_argument('--network', type=str, default='r50', help='backbone network')
    parser.add_argument('--database', type=str, default=None, help='Database saved by postfix .json')
    parser.add_argument('--weight', type=str, default='')
    parser.add_argument('--images', type=str, default=None,
                        help='Run folder contains many images')
    args = parser.parse_args()
    
    with torch.no_grad():
        f = open ('database.json', "r")
        database = json.loads(f.read())

        # Load model arcface + resnet
        recognition = get_model(args.network, fp16=False)
        recognition.load_state_dict(torch.load(args.weight, map_location='cuda'))
        recognition.to('cuda')
        recognition.eval()

        evaluate(args.images, recognition, database)
