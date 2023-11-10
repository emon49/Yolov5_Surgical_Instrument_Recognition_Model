import torch
from matplotlib import pyplot as plt
import numpy as np
import cv2
import os
import easyocr

def get_matches(query, choices, limit):
    from fuzzywuzzy import process
    results = process.extract(query, choices, limit=limit)
    return results


def string_matching_approximation(speech_data):
    #instruments=['Pick','Sort','Sterilize','Scalpel nÂº4','Dissection Clamp','Straight Mayo Scissor','Curved Mayo Scissor']
    instruments=['Scalpel','Dissection Clamp','Mayo Scissor','Place','Please']
    percentages=get_matches(speech_data, instruments,len(instruments))
    max_val=-1
    max_equipment=""
    #print(percentages)
    if percentages[0][1]>50:
        return percentages[0][0]
    
model = torch.hub.load('ultralytics/yolov5', 'custom', path='./best.pt',force_reload=True)
#print(model)
    
os.environ['KMP_DUPLICATE_LIB_OK']='True'
#img = os.path.join('data', 'images', 'bisturi1.jpg')
img = os.path.join('2_Cropprd_Image.jpg')
results = model(img)
#print('\n Indices: - ', results.xyxy)
#print(type(results.xyxy[0][0][0]))
x_mid_points=[]
y_mid_points=[]
fetched_labels=[]

for i in range(len(results.pandas().xyxy[0])):
    x_min = results.pandas().xyxy[0]["xmin"].values[i]
    y_min = results.pandas().xyxy[0]["ymin"].values[i]
    x_max = results.pandas().xyxy[0]["xmax"].values[i]
    y_max = results.pandas().xyxy[0]["ymax"].values[i]
    label = results.pandas().xyxy[0]["name"].values[i]
    
    x_mid=(x_min+x_max)/2
    y_mid=(y_min+y_max)/2
    x_mid_points.append(x_mid)
    y_mid_points.append(y_mid)
    print(i,"    -->>    ",x_mid,y_mid)
    print(label)
#print(x_mid_points)
#print(y_mid_points)

#results.print()
#matplotlib inline 
plt.imshow(np.squeeze(results.render()))
#plt.show()
plt.axis('off')
plt.savefig("output_img.png",bbox_inches="tight")