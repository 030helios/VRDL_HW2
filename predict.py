import os
import argparse
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
import json
# custom module
from util import *
from src.dataset import *
from src.model import EfficientDet

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda")


def get_args():
    ''' get arguments '''
    parser = argparse.ArgumentParser("EfficientDet")
    parser.add_argument("--expname", type=str,
                        default='EXP', help="experiment name")
    parser.add_argument("--best", type=str,
                        default="/HW2model.pth.tar", help="best model")

    args = parser.parse_args()
    args.best = args.expname + args.best

    return args


def main(args):
    torch.cuda.manual_seed(1)

    # load test data
    test_params = {"batch_size": 1,
                   "shuffle": False,
                   "drop_last": False,
                   "num_workers": 2}

    test_set = TestDataset()
    test_loader = DataLoader(test_set, **test_params)

    # define model
    model = EfficientDet(num_classes=10)
    model = model.to(device)
    model = nn.DataParallel(model)

    # load model
    print('loading checkpoint {}'.format(args.best))
    checkpoint = torch.load(args.best)
    args.best_loss = checkpoint['best_loss']
    model.load_state_dict(checkpoint['state_dict'])
    print('loaded checkpoint {}'.format(args.best))
    print('best loss:', args.best_loss)

    # test
    test(model, test_loader, args, [str.split('/')[-1] for str in test_set.x])


def test(model, test_loader, args, ids):
    ''' predict and write json file '''
    model.eval()
    bbox = []
    label = []
    score = []

    with torch.no_grad():
        for i, sample in enumerate(test_loader):
            print(i, '/', len(test_loader), end='     \r')
            image = sample['img'].to(device).float()
            scale = float(sample['scale'])

            # predict
            nms_scores, nms_class, nms_anchors = model([image])

            # rescale the bounding boxes to original size
            nms_scores = nms_scores.cpu().detach().numpy()
            nms_class = nms_class.cpu().detach().numpy()
            nms_anchors = (nms_anchors / scale).cpu().detach().numpy()

            bbox.append(nms_anchors)
            label.append(nms_class)
            score.append(nms_scores)

    # write json
    result_to_json = []
    with open("answer.json", 'w') as File:
        for i, sample in enumerate(test_loader):
            bbs = [list(np.around(b, decimals=13).astype(float))for b in bbox[i]]
            tmp_json = []
            for j, garbage in enumerate(bbs):
                det_box_info = {}
                det_box_info["image_id"] = int(ids[i][:-4])
                det_box_info["score"] = float(score[i][j])
                det_box_info["category_id"] = int(label[i][j]) if label[i][j] else 10
                det_box_info["bbox"] = [bbs[j][1],bbs[j][0], bbs[j][3], bbs[j][2]]
                tmp_json.append(det_box_info)
            result_to_json.extend(sorted(tmp_json, key=lambda d: d["category_id"]))
        # Write the list to answer.json
        json_object = json.dumps(result_to_json, indent=4)
        File.write(json_object)

    return


if __name__ == "__main__":
    args = get_args()
    main(args)
