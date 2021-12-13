import os
import cv2
import argparse
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description='Get mIOU of video sequences')
    parser.add_argument('-i', '--inp_path', type=str, default='groundtruth', required=True, \
                                                        help="Path for the ground truth masks folder")
    args = parser.parse_args()
    return args


def rect_iou(mask1, mask2):
    mask1_area = np.count_nonzero(mask1 == 255)
    mask2_area = np.count_nonzero(mask2 == 255)
    intersection = np.count_nonzero(np.logical_and(mask1==255,  mask2==255))
    union = mask1_area+mask2_area-intersection
    if union == 0: 
        # only happens if both masks are background with all zero values
        iou = 0
    else:
        iou = intersection/union 
    return iou


def main(args):
    # Note: make sure to only generate masks for the evaluation frames mentioned in eval_frames.txt
    # Keep only the masks for eval frames in <inp_path> and not the background (all zero) frames.
    #filenames = os.listdir(args.inp_path)
    ious = []
    groundtruth_rect_file = open(str(args.inp_path)+"/groundtruth_rect.txt","r")
    with open(str(args.inp_path)+"/prediction"+".txt","r") as prediction_rect_file:
        gt = groundtruth_rect_file.readline()
        pr = prediction_rect_file.readline()
        gt = groundtruth_rect_file.readline()
        pr = prediction_rect_file.readline()
        while pr:
            template_rect_gt = str(gt).split(',')
            template_rect_pr = str(pr).split('\t')
            template_x_gt = int(template_rect_gt[0])
            template_y_gt = int(template_rect_gt[1])
            template_w_gt = int(template_rect_gt[2])
            template_h_gt = int(template_rect_gt[3])
            template_x_pr = int(template_rect_pr[0])
            template_y_pr = int(template_rect_pr[1])
            template_w_pr = int(template_rect_pr[2])
            template_h_pr = int(template_rect_pr[3])
            gt_mask = np.zeros((max(template_x_gt+template_w_gt,template_x_pr+template_w_pr),max(template_y_gt+template_h_gt,template_y_pr+template_h_pr)))
            cv2.rectangle(gt_mask, (template_x_gt, template_y_gt), (template_w_gt, template_h_gt), (255, 255, 255), -1)
            pred_mask = np.zeros((max(template_x_gt+template_w_gt,template_x_pr+template_w_pr),max(template_y_gt+template_h_gt,template_y_pr+template_h_pr)))
            cv2.rectangle(pred_mask, (template_x_pr, template_y_pr), (template_w_pr, template_h_pr), (255, 255, 255), -1)
            gt = groundtruth_rect_file.readline()
            pr = prediction_rect_file.readline()
            iou = rect_iou(gt_mask, pred_mask)
            ious.append(iou)
    print("mIOU: %.4f"%(sum(ious)/len(ious)))
    

if __name__ == "__main__":
    args = parse_args()
    main(args)
