import torch
import torch.nn.functional as F
import cv2
import os

from model_rnn import MyModel


#check if on windows OS
windows = False
if os.name == 'nt':
    windows = True

model = MyModel()
model.load_state_dict(torch.load("save/model_rnn12.ckpt"))
model.eval()

img = cv2.imread('steering_wheel_image.jpg',0)
rows,cols = img.shape

smoothed_angle = 0

i = 37000
print("start model 600")
while(cv2.waitKey(10) != ord('q')):
  
    full_image = cv2.imread("driving_dataset/" + str(i) + ".jpg")
    image = cv2.resize(full_image[-150:], (200, 66)) / 255.0
    # Convert the image to PyTorch tensor
    image_tensor = torch.from_numpy(image.transpose((2, 0, 1))).float().unsqueeze(0)
    #print(image_tensor.shape)

    # Predict the steering angle
    degrees = model(image_tensor)[0][0].item() * 180.0 / 3.14159265
    print(str(i)+" Predicted steering angle: " + str(degrees) + " degrees")

    cv2.imshow("frame", full_image)
    
    full_image1 = cv2.imread("result/result/" + str(i) + ".jpg")
    image1 = cv2.resize(full_image[-150:], (200, 66)) / 255.0
    # Convert the image to PyTorch tensor
    image_tensor = torch.from_numpy(image1.transpose((2, 0, 1))).float().unsqueeze(0)
    cv2.imshow("frame_segmentation_image", full_image1)

# Make smooth angle transitions by turning the steering wheel based on the difference of the current angle
# and the predicted angle
    smoothed_angle += 0.2 * pow(abs((degrees - smoothed_angle)), 2.0 / 3.0) * (degrees - smoothed_angle) / abs(degrees - smoothed_angle)
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), -smoothed_angle, 1)
    dst = cv2.warpAffine(img, M, (cols, rows))
    cv2.imshow("steering wheel", dst)
    i += 1

cv2.destroyAllWindows()
