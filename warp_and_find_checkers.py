import cv2
import sys
import numpy as np
import os
import json
#top_left = [71,62]
#top_right = [59, 418]
#bottom_right = [443, 442]
#bottom_left = [438, 29]

def load_preprocessing(img_path, json_file):
    img = cv2.imread(img_path)    
    
    top_left = json_file["canonical_board"]["tl_tr_br_bl"][0]
    top_right = json_file["canonical_board"]["tl_tr_br_bl"][1]
    bottom_right = json_file["canonical_board"]["tl_tr_br_bl"][2]
    bottom_left = json_file["canonical_board"]["tl_tr_br_bl"][3]
    
    
    width = top_right[0] - top_left[0]
    height = bottom_left[1] - top_left[1]
    
    if(height < 0):
        cv2.rotate(img,cv2.ROTATE_90_CLOCKWISE)
        height = bottom_left[1] - top_right[1]
    
    margin_w = width * (0.03)
    margin_h = height * (0.03)
    
    top_left = [top_left[0]-margin_w, top_left[1]-margin_h]
    top_right = [top_right[0]+margin_w, top_right[1]-margin_h]
    bottom_right = [bottom_right[0]+margin_w, bottom_right[1]+margin_h]
    bottom_left = [bottom_left[0]-margin_w, bottom_left[1]+margin_h]
    
    input_ = np.float32([top_left, top_right, bottom_right, bottom_left])
    output = np.float32([[0,0], [width-1,0], [width-1,height-1], [0,height-1]])
    
    matrix = cv2.getPerspectiveTransform(input_,output)
    
    output_image = cv2.warpPerspective(img, matrix,(width,height))
    return output_image

def detect_checkers(img):
    img = cv2.resize(img,(768,759))
    img = cv2.medianBlur(img,5)
    gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit = 10) 
    
    gray_img = clahe.apply(gray_img)
    checkers = cv2.HoughCircles(gray_img,cv2.HOUGH_GRADIENT,1,25,
                            param1=50,param2=30,minRadius=23,maxRadius=40)
    checkers = np.uint16(np.around(checkers))
    
    for i in checkers[0,:]:
        # draw the outer circle
        cv2.circle(img,(i[0],i[1]),i[2],(0,0,255),2)
        # draw the center of the circle
        cv2.circle(img,(i[0],i[1]),2,(0,0,255),3)
        
    cv2.imshow("detected checkers", img)
    cv2.waitKey(0)
    return checkers, img

def save_checkers_count(img,checkers, json_output_path, json_file):
    checkers_count = {"top":[0,0,0,0,0,0,0,0,0,0,0,0],"bottom":[0,0,0,0,0,0,0,0,0,0,0,0]}
    pip_length =  json_file["canonical_board"]["pip_length_to_board_height"] * 759
    pip_shape = [int(768/12),int(pip_length)+15]
    c6 = 60*json_file["canonical_board"]["bar_width_to_checker_width"]
    for i in range(12):
        x = 0
        if i >= 6:
            for c in checkers[0]:
                if c[0] > i*pip_shape[0]  and c[0] < (i+1)*pip_shape[0]  and c[1] > 0 and c[1] < pip_shape[1]:
                    x += 1
                    checkers_count["top"][i] += 1
                    
        else:
            for c in checkers[0]:
                if c[0] > i*pip_shape[0] and c[0] < (i+1)*pip_shape[0]  and c[1] > 0 and c[1] < pip_shape[1]:
                    x += 1
                    checkers_count["top"][i] += 1
            cv2.rectangle(img,(i*pip_shape[0],  pip_shape[1]),((i+1)*pip_shape[0],pip_shape[1]),(255,0,0))
    for i in range(12):
        x = 0
        if i >= 6:
            for c in checkers[0]:
                if c[0] > i*pip_shape[0] and c[0] < (i+1)*pip_shape[0] and c[1] > (759 - pip_length) and c[1] < 759:
                    x += 1
                    checkers_count["bottom"][i] += 1
                    
        else:
            for c in checkers[0]:
                if c[0] > i*pip_shape[0] and c[0] < (i+1)*pip_shape[0]and c[1] > (759 - pip_length) and c[1] < 759:
                    x += 1
                    checkers_count["bottom"][i] += 1
    with open(json_output_path,"w") as file_write:
        json.dump(checkers_count, file_write)

def main(input_dir, output_dir):
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    for image in os.listdir(input_dir):
        if image[-4:] == ".jpg":
            image_path = input_dir + "/" + image
            json_full_name = image_path + ".info.json"
            with open(json_full_name,"r") as file_:
                json_file = json.load(file_)
                img_proc = load_preprocessing(image_path, json_file)
                checkers, out_img  = detect_checkers(img_proc)
                cv2.imwrite(output_dir+'/'+image+".visual_feedback.jpg", out_img)
                json_output_path = output_dir + "/" + image + ".checkers.json"
                save_checkers_count(out_img,checkers, json_output_path, json_file)
                
if __name__ == "__main__":
    if not len(sys.argv) == 3:
        print("usage: python warp_and_find_checkers.py <input_path> <output_path>")
        exit(0)
    
    main(sys.argv[1], sys.argv[2])