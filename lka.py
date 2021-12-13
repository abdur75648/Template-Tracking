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

##################### Lukas Kanade Template Tracking #####################

def jacobian(x_shape, y_shape):
    
    # Clearly dW/dp = d([ x+xP[0]+yP[2]+P[4] , xP[1]+y+yP[3]+P[5] ])/dp # Because Warp Matrix is [[1+p[0], p[2], p[4]], [p[1], 1+p[3], p[5]]])
    # So, dW/dp = [ [x 0 y 0 1 0] ,[0 x 0 y 0 1] ]
    # We do this for all the points on template
    
    x = np.array(list(range(x_shape)))
    y = np.array(list(range(y_shape)))
    x, y = np.meshgrid(x, y) 
    ones = np.ones((y_shape, x_shape))
    zeros = np.zeros((y_shape, x_shape))

    # Stack these 2-D arrays (Each point in x,y,ones,zeroes depicts a point in templates)
    row1 = np.stack((x, zeros, y, zeros, ones, zeros), axis=2)
    row2 = np.stack((zeros, x, zeros, y, zeros, ones), axis=2)
    jacobian = np.stack((row1, row2), axis=2)

    return jacobian


# Main Lukas Kanade Function for a given frame & a given template
def lukas_kanade(img,template,rect,p,dp_thresh= 0.001, max_iter=1000):
    # img = Input image -> [Height X Width]
    # Template = W_img_cropped to be tracked
    # P = Parameters of transformation
    # rect = [[x_top_left,y_top_left],[x_bottom_right,y_bottom_right]]

    template = template[rect[0][1]:rect[1][1], rect[0][0]:rect[1][0]] # Crop main template from image

    iter = 0
    dp_norm =  np.inf
    
    while ((dp_norm >= dp_thresh) and (iter <= max_iter)):
        iter += 1
        """
        if iter%100==0:
            print(f"\nRunning iteration no. {iter}. dp_norm = {dp_norm}")
        """

        # Warp Matrix
        W = np.array([[1+p[0], p[2], p[4]], [p[1], 1+p[3], p[5]]])

        # W(x,p) as per lecture notes
        W_img = cv2.warpAffine(img, W, (img.shape[1],img.shape[0]), flags=cv2.INTER_CUBIC)
        W_img_cropped = W_img [rect[0][1]:rect[1][1], rect[0][0]:rect[1][0]]

        dif = template.astype(int) - W_img_cropped.astype(int) # Same sa T(x) - I(W(x,p))

        ### Calculating warped gradients # ∇I
        # img gradient
        grad_x = cv2.Sobel(src=np.float32(img), ddepth=cv2.CV_64F, dx=1, dy=0, ksize=5)
        grad_y = cv2.Sobel(src=np.float32(img), ddepth=cv2.CV_64F, dx=0, dy=1, ksize=5)
        # Warping the gradient & cropping it to get ∇I evaluated at W(x,p) in roi
        I_x = cv2.warpAffine(grad_x,W,(img.shape[1],img.shape[0]), flags=cv2.INTER_CUBIC)[rect[0][1]:rect[1][1], rect[0][0]:rect[1][0]]
        I_y = cv2.warpAffine(grad_y,W,(img.shape[1],img.shape[0]), flags=cv2.INTER_CUBIC)[rect[0][1]:rect[1][1], rect[0][0]:rect[1][0]]
        # Stack them to get the ∇I
        grad_I = np.stack((I_x,I_y),axis=2)
        grad_I = np.expand_dims((grad_I),axis=2)

        # Get jacobian of thee warping for template size = dW/dp
        jacobian_of_warping = jacobian(template.shape[1],template.shape[0])

        # Calculate steepest descent = ∇I dW/dp
        dW_by_dp = np.matmul(grad_I, jacobian_of_warping)
        dW_by_dp_T = np.transpose(dW_by_dp, (0, 1, 3, 2))


        ### Computing dp      
        dif = dif.reshape((template.shape[0],template.shape[1], 1, 1))
        main_term = (dW_by_dp_T * dif).sum((0,1))

        # Compute H
        H = np.matmul(dW_by_dp_T, dW_by_dp).sum((0,1))

        # calculate dp
        dp = np.matmul(np.linalg.pinv(H), main_term).reshape((-1))

        p += dp
            
        dp_norm = np.linalg.norm(dp)
    
    # Return parameter of transformation
    return p

# Lucas Kanade Tracking
def lka_tt(args):
    with open(str(args.inp_path)+"/groundtruth_rect.txt","r") as groundtruth_rect_file:
        gts = groundtruth_rect_file.readlines()
        template_rect = str(gts[0]).split('\t')
        template_x = int(template_rect[0])
        template_y = int(template_rect[1])
        template_w = int(template_rect[2])
        template_h = int(template_rect[3])
    
    rect = np.array([[template_x,template_y], [template_x+template_w,template_y+template_h]]) 

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
                frame = cv2.GaussianBlur(frame, (5, 5), 0)

                if(file_number==1):
                    template = frame # [template_y:template_y+template_h, template_x:template_x+template_w]
                    # template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY) # Already fram is Grayscale
                    # template = cv2.GaussianBlur(template, (5, 5), 0)
                frame = (frame * (np.mean(template)/np.mean(frame)).astype(float) ) # Normalise frame as per template
                # Initialise W as Identity means parameters P1-P6 as zero
                p = np.zeros(6)

                p = lukas_kanade(frame,template,rect,p)
                Warping_matrix_final = np.array([[1 + p[0], p[2], p[4]], [p[1], 1 + p[3], p[5]]])

                tl_homogeneous = np.array([rect[0][0], rect[0][1], 1])
                br_homogeneous = np.array([rect[1][0], rect[1][1], 1])
                
                tl_homogeneous_new = (np.matmul(Warping_matrix_final,tl_homogeneous)).astype(int)
                br_homogeneous_new = (np.matmul(Warping_matrix_final,br_homogeneous)).astype(int)
                cv2.rectangle(original_frame, tuple(tl_homogeneous_new), tuple(br_homogeneous_new), (0, 0, 255), 2)
                cv2.putText(original_frame, 'Detected', (tl_homogeneous_new[0], tl_homogeneous_new[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 2)

                # Drawing Ground Truth
                gt = gts[file_number-1]
                gt = str(gt).split('\t')
                cv2.rectangle(original_frame, tuple([int(gt[0]),int(gt[1])]), tuple([int(gt[0]) + int(gt[2]),int(gt[1]) + int(gt[3])]), (0, 255, 0), 2)
                cv2.putText(original_frame, 'Ground Truth', (int(gt[0]), int(gt[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                out_file.write(f'{tl_homogeneous_new[0]}\t{tl_homogeneous_new[1]}\t{(br_homogeneous_new[0]-tl_homogeneous_new[0])}\t{(br_homogeneous_new[1]-tl_homogeneous_new[1])}\n')
                cv2.imwrite((str(args.inp_path)+"/vis/"+str(file_number)+".jpg"),original_frame)
                cv2.imshow('Tracked Image', original_frame)
                if cv2.waitKey(1) & 0xFF==ord('q'):
                    print("Done")
                    break

                
        # Closes all the frames
        cv2.destroyAllWindows()
    out_file.close()
    pass

if __name__ == "__main__":
    args = parse_args()
    lka_tt(args)
