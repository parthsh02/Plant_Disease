import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

filepath = '/Users/parthsharma/Important/Plant-Leaf-Disease-Prediction-main/model_inception.h5'
model = load_model(filepath)
print(model)

print("Model Loaded Successfully")

tomato_plant = cv2.imread('/Users/parthsharma/Important/Plant-Leaf-Disease-Prediction-main/Dataset/train/Orange___Haunglongbing_(Citrus_greening)/4115edd8-1516-4d05-8198-e3b4dbcf9af3___CREC_HLB 5518.JPG')
test_image = cv2.resize(tomato_plant, (224,224)) # load image 
  
test_image = img_to_array(test_image)/255 # convert image to np array and normalize
test_image = np.expand_dims(test_image, axis = 0) # change dimention 3D to 4D
  
result = model.predict(test_image) # predict diseased palnt or not
  
pred = np.argmax(result, axis=1)
print(pred)

if pred==0:
    print( "Apple Cedar Apple Rust")

elif pred==1:
    print( "Cherry_(including_sour)___Powdery_mildew")

elif pred==2:
    print("Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot")


elif pred==3:
    print( "Corn_(maize)___Common_rust_")
 
elif pred==4:
    print( "Corn_(maize)___Northern_Leaf_Blight")
    
elif pred==5:
    print( "Grape___Black_rot")
    
elif pred==6:
    print( "Grape___Esca_(Black_Measles)")    

elif pred==7:
    print( "Orange___Haunglongbing_(Citrus_greening)")
    
elif pred==8:
    print( "Peach___Bacterial_spot")

elif pred==9:
    print( "Potato___Early_blight")
    
elif pred==10:
    print( "Potato___Late_blight") 
    
elif pred==11:
    print( "Strawberry___Leaf_scorch_")
          
elif pred==12:
    print( "Tomato - Bacterial_spot")        
      
elif pred==13:
    print("Tomato - Early Blight Disease")       
        
elif pred==14:
    print("Tomato - Late Blight Disease")
       
elif pred==15:
    print("Tomato - Leaf Mold Disease")
        
elif pred==16:
    print("Tomato - Septoria Leaf Spot Disease")        
        
elif pred==17:
      print("Tomato - Tomoato Yellow Leaf Curl Virus Disease")
      
elif pred==18:
      print("Tomato - Tomato Mosaic Virus Disease")
        

        
