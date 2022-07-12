import argparse
import cv2
import torch
import json
import os
import matplotlib.pyplot as plt
import numpy as np
from re import L
from pathlib import Path
from mtcnn.mtcnn import MTCNN
from backbones import get_model

# Angle between two vector : Cosine similarity
def cosine_similarity(emb1, emb2):
    return np.sum((emb1 * emb2)) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

# Euclid distance similarity
def distance_similarity(emb1, emb2):
    return np.sum((emb1-emb2)**2)**0.5


def inference(detect_face, recognition, img, database):

    # Detect all bounding box face humans
    results_bbox = detect_face.detect_faces(img)

    list_bbox = []
    # Save list bbox human 
    for result in results_bbox:
        list_bbox.append(result['box'])

    # Recognition object bbox in image
    for index, bbox in enumerate(list_bbox):
        cropimg = cv2.resize(img[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2], :], (112, 112))
        cropimg = cv2.cvtColor(cropimg, cv2.COLOR_BGR2RGB)
        cropimg = np.transpose(cropimg, (2, 0, 1))
        cropimg = torch.from_numpy(cropimg).unsqueeze(0).float()
        cropimg.div_(255).sub_(0.5).div_(0.5)
        
        emb_vector = recognition(cropimg).numpy()
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
        
        print(data[maxkey_similarity])
        # Print Text and rectange bbox in image
        text = ''
        if data[maxkey_similarity] > args.threshold:
            text += f'Label %s ' %maxkey_similarity
            text += str(round(data[maxkey_similarity], 2))
        else:
            text += 'Unknown'
        print(text)


        cv2.rectangle(img, (bbox[0] - 5, bbox[1] - 10), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (0, 255, 0), 2)
        cv2.putText(img, text, (bbox[0], bbox[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    return img


def evalimage(detect_face, recognition, database, path: str, save_path: str=None):
    if os.path.exists(path):
        img = cv2.imread(path)
        
        img = inference(detect_face, recognition, img, database)
        
        if save_path is None:
            plt.imshow(img[:, :, (2, 1, 0)])
            plt.title(path)
            plt.show()
        else:
            cv2.imwrite(img, save_path)

    else:
        raise FileNotFoundError


def evalimages(detect_face, recognition, database, input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    for p in Path(input_folder).glob('*'): 
        path = str(p)
        name = os.path.basename(path)
        name = '.'.join(name.split('.')[:-1]) + '.png'
        out_path = os.path.join(output_folder, name)

        evalimage(detect_face, recognition, path, out_path)
        print(path + ' -> ' + out_path)
    print('Done.')


def evalwebcam(detect_face, recognition, database):

    vid = cv2.VideoCapture(0)

    while True:
        # Capture the video frame
        ret, frame = vid.read()
        # Display the resulting frame
        # Detect all bounding box face humans
        frame = inference(detect_face, recognition, frame, database)
        
        cv2.imshow('frame', frame)
      
        # the 'q' button is set as the
        # quitting button you may use any
        # desired button of your choice
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

def evalvideo(detect_face, recognition, database, path, out_path: str=None):
    
    vid = cv2.VideoCapture(path)
    
    if not vid.isOpened():
        print('Could not open video "%s"' % path)
        exit(-1)

    target_fps   = round(vid.get(cv2.CAP_PROP_FPS))
    frame_width  = round(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = round(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), target_fps, (frame_width, frame_height))

    while True:
        # Capture the video frame
        ret, frame = vid.read()
        # Display the resulting frame
        # Detect all bounding box face humans
        frame = inference(detect_face, recognition, frame, database)

        out.write(frame)



def run(detect_face, recognition, database, args):
    if args.image is not None:
        if ':' in args.image:
            inp, out = args.image.split(':')
            evalimage(detect_face, recognition, database, inp, out)
        else:
            evalimage(detect_face, recognition, database, args.image)
        return
    elif args.images is not None:
        inp, out = args.images.split(':')
        evalimages(detect_face, recognition, database, inp, out)
        return
    elif args.video is not None:
        if ':' in args.video:
            inp, out = args.video.split(':')
            evalvideo(detect_face, recognition, database, inp, out)
        else:
            evalvideo(detect_face, recognition, database, args.video)
    elif args.webcam is not None:
        evalvideo(detect_face, recognition, database)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch ArcFace Evaluation')
    parser.add_argument('--network', type=str, default='r50', help='backbone network')
    parser.add_argument('--weight', type=str, default='')
    parser.add_argument('--threshold', type=float, default=0.45)
    parser.add_argument('--image', type=str, default=None,
                        help='Run one image')
    parser.add_argument('--images', type=str, default=None,
                        help='Run folder contains many images')
    parser.add_argument('--video', type=str, default=None,
                        help='Run video .mp4, .avi, v.v')
    parser.add_argument('--webcam', type=int, default=0)
    args = parser.parse_args()
    

    with torch.no_grad():
        # Load databease
        f = open ('r100_database.json', "r")
        database = json.loads(f.read())

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Load model MTCNN
        print("Loading model MTCNN ....")
        detect_face = MTCNN()
        print("Loaded model MTCNN")

        # Load model arcface + resnet
        recognition = get_model(args.network, fp16=False)
        recognition.load_state_dict(torch.load(args.weight, map_location=device))
        recognition.to(device)
        recognition.eval()

        # evaluate
        run(detect_face, recognition, database, args)