import os,shutil
import cv2
import argparse
import numpy as np
import copy
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description='Get mIOU of video sequences')
    parser.add_argument('-i', '--inp_path', type=str, default='BlurCar2', help="Path for the root of input folder. (BlurCar2/Bolt/Liquor)")
    args = parser.parse_args()
    return args

def ncc_tt(args):
    with open(str(args.inp_path)+"/groundtruth_rect.txt","r") as groundtruth_rect_file:
        gts = groundtruth_rect_file.readlines()
        template_rect = str(gts[0]).split('\t')
        template_x = int(template_rect[0])
        template_y = int(template_rect[1])
        template_w = int(template_rect[2])
        template_h = int(template_rect[3])

    if os.path.exists(str(args.inp_path)+"/vis"):
        shutil.rmtree(str(args.inp_path)+"/vis")
    os.mkdir(str(args.inp_path)+"/vis")
    if os.path.exists(str(args.inp_path)+"/prediction.txt"):
        os.remove(str(args.inp_path)+"/prediction.txt")
    out_file = open(str(args.inp_path)+"/prediction.txt","a",encoding="utf-8")
    for dirs,subdirs,files in os.walk(args.inp_path+"/img"):
        files = sorted(files)
        for file in tqdm(files):
            if file.endswith('.png') or file.endswith('.jpg') or file.endswith('.jpeg'):
                file_number = int(str(file)[:-4])
                input_image = os.path.join(args.inp_path+"/img",file)
                frame = cv2.imread(input_image)
                original_frame = copy.deepcopy(frame)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                done = None
                # frame = cv2.GaussianBlur(frame, (5, 5), 0)
                # frame = (frame * (np.mean(template)/np.mean(frame)).astype(float) ) # Normalise frame as per template
                if(file_number==1):
                    template = frame[template_y:template_y+template_h, template_x:template_x+template_w]
                    template = cv2.Canny(template, 50, 200)
                    # template = cv2.GaussianBlur(template, (5, 5), 0)
                sList = np.linspace(0.2, 1.0, 20)[::-1]
                for s in sList:
                    scaled = cv2.resize(frame, (0,0), fx=int(frame.shape[1] * s)/frame.shape[1], fy=int(frame.shape[1] * s)/frame.shape[1])
                    r = frame.shape[1] / float(scaled.shape[1])
                    if scaled.shape[1] < template_w or scaled.shape[0] < template_h:
                        break
                    frameEdge = cv2.Canny(scaled, 50, 200)
                    final = cv2.matchTemplate(frameEdge, template, cv2.TM_CCOEFF_NORMED)
                    (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(final)
                    if done is None or maxVal > done[0]:
                        done = (maxVal, maxLoc, r)
                (_, maxLoc, r) = done
                # Drawing Prediction
                cv2.rectangle(original_frame , (int(maxLoc[0] * r), int(maxLoc[1] * r)), (int((maxLoc[0] + template_w) * r), int((maxLoc[1] + template_h) * r)), (255, 0, 0), 2)
                cv2.putText(original_frame , 'Prediction', (int(maxLoc[0] * r), int(maxLoc[1] * r)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, ( 255, 0, 0), 2)
                # Drawing Ground Truth
                gt = gts[file_number-1]
                gt = str(gt).split('\t')
                cv2.rectangle(original_frame , tuple([int(gt[0]),int(gt[1])]), tuple([int(gt[0]) + int(gt[2]),int(gt[1]) + int(gt[3])]), (0, 255, 0), 2)
                cv2.putText(original_frame , 'Ground Truth', (int(gt[0]), int(gt[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                cv2.imwrite((str(args.inp_path)+"/vis/"+str(file_number)+".jpg"),original_frame)
                cv2.imshow('TemplateTracker',original_frame )
                out_file.write(f'{int(maxLoc[0] * r)}\t{int(maxLoc[1] * r)}\t{template_w}\t{template_h}\n')
                #print(file)
                # Press Q on keyboard to  exit
                if cv2.waitKey(1) & 0xFF==ord('q'):
                    print("Done")
                    break
        # Closes all the frames
        cv2.destroyAllWindows()
    out_file.close()
    pass

if __name__ == "__main__":
    args = parse_args()
    ncc_tt(args)
