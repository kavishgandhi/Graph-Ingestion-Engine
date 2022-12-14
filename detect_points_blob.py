from collections import defaultdict, Counter
from turtle import left, width
import cv2
import numpy as np
from colorthief import ColorThief
import regex
from PIL import Image
import easyocr

# detect and remove axes from plot
class usingBlobs():
    def __init__(self) -> None:
        self.keypoints = []
        self.center_points = []

    def remove_axes(self, img):
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        th, img_gray_th_otsu = cv2.threshold(img_gray, 128, 255, cv2.THRESH_OTSU)
        edges = cv2.Canny(img_gray_th_otsu, 50, 150)
        rho = 1  # distance resolution in pixels of the Hough grid
        theta = np.pi / 180  # angular resolution in radians of the Hough grid
        threshold = 15  # minimum number of votes (intersections in Hough grid cell)
        min_line_length = 50  # minimum number of pixels making up a line
        max_line_gap = 20  # maximum gap in pixels between connectable line segments
        line_image = np.copy(img) * 0  # creating a blank to draw lines on
        # Run Hough on edge detected image
        # Output "lines" is an array containing endpoints of detected line segments
        lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),
                            min_line_length, max_line_gap)
        xaxis_len = 0
        xaxis_idx = -1
        yaxis_len = 0
        y_axis_idx = -1
        for idx,line in enumerate(lines):
            for x1,y1,x2,y2 in line:
                if x1==x2: #vertical line
                    if np.abs(y1-y2)>yaxis_len:
                        yaxis_len = np.abs(y1-y2)
                        yaxis_idx = idx
                if y1==y2: #vertical line
                    if np.abs(x1-x2)>xaxis_len:
                        xaxis_len = np.abs(x1-x2)
                        xaxis_idx = idx
        xx1,xy1,xx2,xy2 = lines[xaxis_idx][0]
        yx1,yy1,yx2,yy2 = lines[yaxis_idx][0]
        _ = cv2.line(line_image,(xx1,xy1),(xx2,xy2),(255,255,255),3)
        _ = cv2.line(line_image,(yx1,yy1),(yx2,yy2),(255,255,255),3)

        gray_line_image = cv2.cvtColor(line_image, cv2.COLOR_BGR2GRAY)
        indices = np.where(gray_line_image==[255])
        coordinates = np.asarray(list(zip(indices[0], indices[1])))
        for i in coordinates:
            img[i[0],i[1]] = 255
        return img

    # extract colors from image
    # def extract_colors(self, img, result):
    #     width, height, _ = img.shape
    #     colors_dict = defaultdict(int)
    #     for i in range(width):
    #         for j in range(height):
    #             colors_dict[tuple(img[i,j])]+=1
    #     colors_dict = dict(sorted(colors_dict.items(), key=lambda x:x[1], reverse=True))
    #     new_colors_dict = defaultdict(int)
    #     for k, v in colors_dict.items():
    #         if k[0] != k[1] and k[1] != k[2] and v>10:
    #             new_colors_dict[k] = v
    #     for color in new_colors_dict.keys():
    #         img_clone = img.copy()
    #         for i in range(width):
    #             for j in range(height):
    #                 if (img_clone[i,j]!=color).all():
    #                     img_clone[i,j] = (0)
    #         result = self.blob_detection(img_clone, result)
    #     return result

    # blob detection
    def blob_detection(self, im):
        detector = cv2.SimpleBlobDetector_create()
        img_clone = im.copy()
        while True:
            kp = detector.detect(img_clone)
            if len(kp)==0:
                break
            self.keypoints.append(kp)
            for i in kp:
                x, y = round(i.pt[0]), round(i.pt[1])
                self.center_points.append((x,y))
                img_clone[y,x] = (0,0,0)
                cv2.circle(img_clone, (x, y), round(i.size/2), (0,0,0), -1)
            width, height, _ = img_clone.shape
            for i in range(width):
                for j in range(height):
                    if (img_clone[i, j] == (0,0,0)).all():
                        img_clone[i,j]=(255,255,255)
            # result = cv2.add(result, im_with_keypoints) #if running extract_colors+blob_detection
            # break
        
        im_with_keypoints = im.copy()
        for i in self.keypoints:
            im_with_keypoints = cv2.drawKeypoints(im_with_keypoints, i, np.array([]), (0,0,0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        op_img = im_with_keypoints #if running only blob_detection
        return op_img

if __name__=='__main__':
    blobDetection = usingBlobs()
    img_name = '900.png'
    img = cv2.imread(img_name)
    reader = easyocr.Reader(['en'])
    data = reader.readtext(img)
    filtered_data = []
    detected_texts = []
    for i in data:
        if regex.search(r'\d', i[1]):
            continue
        else:
            filtered_data.append(i)
    for (bbox,text,_) in filtered_data:
        (tl, tr, br, bl) = bbox
        tl = (int(tl[0])-18, int(tl[1]))
        tr = (int(tr[0]), int(tr[1]))
        br = (int(br[0]), int(br[1]))
        bl = (int(bl[0])-18, int(bl[1]))
        cv2.rectangle(img, tl, br, (255,255,255), -1)
        detected_texts.append(text)
    img_ra = blobDetection.remove_axes(img)
    result = blobDetection.blob_detection(img_ra)
    for i in blobDetection.center_points:
        cv2.circle(result, (i[0],i[1]), 2, (0,0,0), -1)
    cv2.imshow("result", result)
    cv2.waitKey(0)
    legend_data = []
    for text in detected_texts:
        text_ = text.replace(" ","")
        if regex.search(r'(?:title){d<=1}', text_, regex.ENHANCEMATCH) or regex.search(r'(?:label){d<=1}', text_, regex.ENHANCEMATCH):
            continue
        else:
            legend_data.append(text)
    # print(legend_data)
    
    
    
    
    
# OLD CODE DO NOT TOUCH    
# pil_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# pil_image = Image.fromarray(pil_image)
# data = do_ocr_trial.get_table(pil_image)
# height, width, _ = img.shape
# x, y, w, h = data.left.values.tolist(), data.top.values.tolist(), data.width.values.tolist(), data.height.values.tolist()
# for i, j, k, l in zip(x, y, w, h):
#     cv2.rectangle(img, (i, j+l), (i+k, j), color=(0,0,255), thickness=1)
# n_boxes = len(data['char'])    
# for i in range(n_boxes):
#     (text,x1,y2,x2,y1) = (data['char'][i],data['left'][i],data['top'][i],data['right'][i],data['bottom'][i])
#     cv2.rectangle(img, (x1,height-y1), (x2,height-y2) , (255,255,255), -1)
# for i in blobDetection.center_points:
#     cv2.circle(img, (i[0],i[1]), 2, (0,0,0), -1)
# cv2.imshow("result", img)
# cv2.waitKey(0)