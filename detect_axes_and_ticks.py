import cv2
import matplotlib.pyplot as plt
import numpy as np
# %matplotlib inline
# from google.colab.patches import cv2_imshow
import keras_ocr
from scipy.spatial import KDTree
from sklearn.linear_model import RANSACRegressor


def read_img(img_path):
  #"/content/drive/MyDrive/Graph Ingestion Engine/Scatter_plots/875.png"
  img = cv2.imread(img_path)
  cv2.imshow("Image",img)
  cv2.waitKey(0)

  return img

def img_to_binary(img):
  # Convert image to grayscale and binary
  img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  th, img_gray_th_otsu = cv2.threshold(img_gray, 128, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
  plt.imshow(img_gray_th_otsu,cmap="Greys")
  return img_gray, img_gray_th_otsu

"""## Axis Detection"""

def detect_edges(img_gray_th_otsu):
  #Edge detection
  edges = cv2.Canny(img_gray_th_otsu, 50 , 100)
  plt.imshow(edges,cmap="Greys")
  plt.show()
  return edges

def line_detection(edges):
  #Line detection using Hough Transform
  rho = 1  # distance resolution in pixels of the Hough grid
  theta = np.pi / 180  # angular resolution in radians of the Hough grid
  threshold = 15  # minimum number of votes (intersections in Hough grid cell)
  min_line_length = 50  # minimum number of pixels making up a line
  max_line_gap = 20  # maximum gap in pixels between connectable line segments

  # Run Hough on edge detected image
  # Output "lines" is an array containing endpoints of detected line segments
  lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),
                      min_line_length, max_line_gap)
  lines = np.array(lines)
  lines = np.squeeze(lines)
  return lines

def find_axes(img,lines):
  line_image = np.copy(img) * 0 + 255  # creating a blank to draw lines on
  #Find longest horizontal and vertical lines as x and y axes
  xaxis_len = 0
  xaxis_idx = -1
  yaxis_len = 0
  y_axis_idx = -1
  for idx,line in enumerate(lines):
    x1,y1,x2,y2=line
    if x1==x2: #vertical line
        if np.abs(y1-y2)>yaxis_len:
            yaxis_len = np.abs(y1-y2)
            yaxis_idx = idx
    if y1==y2: #horizontal line
        if np.abs(x1-x2)>xaxis_len:
            xaxis_len = np.abs(x1-x2)
            xaxis_idx = idx

  xaxis = lines[xaxis_idx]
  yaxis = lines[yaxis_idx]

  xx1,xy1,xx2,xy2 = lines[xaxis_idx]
  yx1,yy1,yx2,yy2 = lines[yaxis_idx]
  _ = cv2.line(line_image,(xx1,xy1),(xx2,xy2),(0,0,0),5)
  _ = cv2.line(line_image,(yx1,yy1),(yx2,yy2),(0,0,0),5)

  plt.imshow(line_image, cmap="gray")
  plt.show()
  return xaxis,yaxis


def detect_x_axis_ticks(img_gray,xaxis):
  """x-axis tick detection"""

  xx1,xy1,xx2,xy2 = xaxis
  x_axis_image_segment = img_gray[xy1-6:xy1+8,xx1:xx2]
  cv2.imshow("X-axis segment",x_axis_image_segment)
  cv2.waitKey(0)

  x_axis_image_segment_bin = np.where(x_axis_image_segment==0,1,0)
  x_axis_profile = np.sum(x_axis_image_segment_bin,axis=0)

  x_tick_idx = np.where(x_axis_profile>4)[0]
  x_tick_idx = x_tick_idx + xx1

  x_ticks_len = len(x_tick_idx)
  x_tick_coords = np.vstack((x_tick_idx,[xy1]*x_ticks_len)).T
  return x_tick_coords

def detect_y_axis_ticks(img_gray,yaxis):
  """Y-Axis tick detection"""

  yx1,yy1,yx2,yy2 = yaxis

  y_axis_image_segment = img_gray[yy2:yy1,yx1-6:yx2+8]
  cv2.imshow("Y-axis segment",y_axis_image_segment)
  cv2.waitKey(0)

  y_axis_image_segment_bin = np.where(y_axis_image_segment==0,1,0)
  y_axis_profile = np.sum(y_axis_image_segment_bin,axis=1)

  y_tick_idx = np.where(y_axis_profile>4)[0]
  y_tick_idx += yy2

  y_ticks_len = len(y_tick_idx)
  y_tick_coords = np.vstack(([yx1]*y_ticks_len,y_tick_idx)).T
  return y_tick_coords

def visualize_ticks(img, x_tick_coords,y_tick_coords):
  """## Visualizing ticks"""

  img2 = img.copy()
  for tick in x_tick_coords:
    img2 = cv2.circle(img2, (tick[0],tick[1]), 2, [0,0,255], 2)
  for tick in y_tick_coords:
    img2 = cv2.circle(img2, (tick[0],tick[1]), 2, [0,0,255], 2)

  cv2.imshow("Image with ticks",img2)
  cv2.waitKey(0)


def text_recognition(img_path):
  """# Text Detection"""

  # keras-ocr will automatically download pretrained
  # weights for the detector and recognizer.
  pipeline = keras_ocr.pipeline.Pipeline()

  images = [keras_ocr.tools.read(img_path), keras_ocr.tools.read(img_path)]

  # Each list of predictions in prediction_groups is a list of
  # (word, box) tuples.
  prediction_groups = pipeline.recognize(images)

  # Plot the predictions
  fig, axs = plt.subplots(nrows=1, figsize=(20, 20))
  # for ax, image, predictions in zip(axs, images[0], prediction_groups[0]):
  keras_ocr.tools.drawAnnotations(image=images[0], predictions=prediction_groups[0], ax=axs)

  predicted_text = np.array([pred[0] for pred in prediction_groups[0]])
  predicted_text_center = np.array([np.mean(pred[1],axis=0) for pred in prediction_groups[0]])

  digit_idx = [idx for idx,text in enumerate(predicted_text) if text.isdigit()]

  predicted_digits = predicted_text[digit_idx]

  predicted_digit_center = predicted_text_center[digit_idx]
  return predicted_digits, predicted_digit_center


def match_tick_to_digits(predicted_digit_center, x_tick_coords, y_tick_coords):
  """Matching ticks and digits"""

  tree = KDTree(predicted_digit_center)
  x_digit_matches = tree.query(x_tick_coords)
  y_digit_matches = tree.query(y_tick_coords)
  return x_digit_matches, y_digit_matches

def filter_tick_matches_by_dist(x_digit_matches, y_digit_matches, x_tick_coords, y_tick_coords, predicted_digits):
  """Filter out matches that are too far"""

  x_digit_dist,x_digit_idx = x_digit_matches

  y_digit_dist,y_digit_idx = y_digit_matches

  tol = 2

  mean_x_dist = np.mean(x_digit_dist)
  mean_y_dist = np.mean(y_digit_dist)

  bad_x_matches = np.where(x_digit_dist> mean_x_dist+tol)[0]
  bad_y_matches = np.where(y_digit_dist> mean_y_dist+tol)[0]

  x_digit_idx = np.delete(x_digit_idx,bad_x_matches)
  y_digit_idx = np.delete(y_digit_idx,bad_y_matches)

  x_tick_coords = np.delete(x_tick_coords,bad_x_matches,axis=0)
  y_tick_coords = np.delete(y_tick_coords,bad_y_matches,axis=0)

  x_digit = np.array(predicted_digits[x_digit_idx],dtype="float")
  x_digit_coords = np.vstack((x_digit,np.zeros(len(x_digit)))).T
  y_digit = np.array(predicted_digits[y_digit_idx],dtype="float")
  y_digit_coords = np.vstack((np.zeros(len(y_digit)),y_digit)).T

  return x_digit_coords,y_digit_coords,x_tick_coords,y_tick_coords

"""### RANSAC Regression"""


def plot_ransac_regression(reg,X,y):
  #
  # Get the Inlier mask; Create outlier mask
  #
  inlier_mask = reg.inlier_mask_
  outlier_mask = np.logical_not(inlier_mask)
  #
  # Create scatter plot for inlier datset
  #
  plt.figure(figsize=(8, 8))
  plt.scatter(X[inlier_mask], y[inlier_mask],
              c='steelblue', edgecolor='white',
              marker='o', label='Inliers')
  #
  # Create scatter plot for outlier datset
  #
  plt.scatter(X[outlier_mask], y[outlier_mask],
              c='limegreen', edgecolor='white',
              marker='s', label='Outliers')
  #
  # Draw the best fit line
  #
  line_X = np.arange(3, 500, 1)
  line_y_ransac = reg.predict(line_X[:, np.newaxis])
  plt.plot(line_X, line_y_ransac, color='black', lw=2)
  plt.xlabel('pixel coords', fontsize=15)
  plt.ylabel('graph coords', fontsize=15)
  plt.legend(loc='upper left', fontsize=12)
  plt.show()

def x_axis_regression(x_tick_coords,x_digit_coords):
  reg_x = RANSACRegressor(min_samples=2,max_trials=100,
                          loss='absolute_loss', random_state=42,
                          residual_threshold=10).fit(x_tick_coords[:,0].reshape(-1,1), x_digit_coords[:,0].reshape(-1,1))

  plot_ransac_regression(reg_x,x_tick_coords,x_digit_coords)
  return reg_x

def y_axis_regression(y_tick_coords,y_digit_coords):
  reg_y = RANSACRegressor(min_samples=3,max_trials=100,
                          loss='absolute_loss', random_state=42,
                          residual_threshold=10).fit(y_tick_coords[:,1].reshape(-1,1), y_digit_coords[:,1].reshape(-1,1))

  plot_ransac_regression(reg_y,y_tick_coords,y_digit_coords)
  return reg_y


def run(img_path):
  img = read_img(img_path)
  img_gray, img_bin = img_to_binary(img)
  edges = detect_edges(img_bin)
  lines = line_detection(edges)
  xaxis,yaxis = find_axes(img,lines)

  x_tick_coords = detect_x_axis_ticks(img_gray,xaxis)
  y_tick_coords = detect_y_axis_ticks(img_gray,yaxis)

  visualize_ticks(img, x_tick_coords,y_tick_coords)

  predicted_digits, predicted_digit_center = text_recognition(img_path)
  x_digit_matches, y_digit_matches = match_tick_to_digits(predicted_digit_center, x_tick_coords, y_tick_coords)
  x_digit_coords,y_digit_coords,x_tick_coords,y_tick_coords = filter_tick_matches_by_dist(x_digit_matches, y_digit_matches, x_tick_coords, y_tick_coords, predicted_digits)

  reg_x = x_axis_regression(x_tick_coords,x_digit_coords)
  reg_y = y_axis_regression(y_tick_coords,y_digit_coords)

  return reg_x, reg_y


if __name__=='__main__':
  reg_x,reg_y = run()




