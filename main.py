from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import io
 
app = FastAPI()
 
# Load the pre-trained model
model = load_model('tensor_v1.h5')
categories = ['agricultural', 'airplane', 'baseballdiamond', 'beach', 'buildings', 'chaparral', 'denseresidential', 'forest', 'freeway', 'golfcourse', 'harbor', 'intersection', 'mediumresidential', 'mobilehomepark', 'overpass', 'parkinglot', 'river', 'runway', 'sparseresidential', 'storagetanks', 'tenniscourt']
 
@app.get("/")
async def root():
    return {"message": "Welcome to the image classification API!"}
 
# Helper function to preprocess the image
def preprocess_image(image_bytes, target_size):
    # Load the image from bytes
    img = Image.open(io.BytesIO(image_bytes))
    # Resize the image
    img = img.resize(target_size)
    # Convert the image to array
    img_array = image.img_to_array(img)
    # Rescale the image
    img_array = img_array / 255.0
    # Expand dimensions to match the input shape (1, IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array
 
# Endpoint to accept image uploads and make predictions
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    processed_image = preprocess_image(contents, target_size=(128, 128))  # Adjust target_size as per your model
    prediction = model.predict(processed_image)
   
    # Convert prediction to class label
    # predicted_class = np.argmax(prediction, axis=1)
    predicted_variables = []
 
    for index in range(len(categories)):
        # Convert np.float64 to native Python float
        prediction_value = float(prediction[0][index])
       
        # Round the float to three decimal places
        rounded_value = round(prediction_value, 3)
       
        print(rounded_value)  # Print the rounded value
       
        if rounded_value > 0.05:
            predicted_variables.append(categories[index])
 
    return JSONResponse(content={"prediction": predicted_variables})