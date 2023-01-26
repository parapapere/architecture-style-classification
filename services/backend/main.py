from io import BytesIO
from PIL import Image
from fastapi import FastAPI
from fastapi import File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import get_file
from tensorflow.keras.utils import load_img
from tensorflow.keras.utils import img_to_array
from tensorflow import expand_dims
from tensorflow.nn import softmax
from numpy import argmax
from numpy import max
from numpy import array
from json import dumps
from uvicorn import run
import os
import numpy as np

app = FastAPI()

origins = ["*"]
methods = ["*"]
headers = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=methods,
    allow_headers=headers
)

model_dir = "model.h5"
model = load_model(model_dir)

twety_first_century = array(['Blobitecture',  'Deconstructivism',  'Eco-architecture',
                            'Neo-futurism architecture',  'Postmodern architecture'])

baroque = array(['Andean Baroque Architecture',  'Earthquake Baroque Architecture',  'Russian Baroque Architecture',
                 'Baroque architecture', 'Rococo Architecture', 'Sicilian Baroque Architecture'])

classical = array(['Achaemenid architecture',  'Ancient Egyptian architecture',
                  'Herodian architecture',  'Roman Classical architecture'])

early_christian_medieval = array(['Byzantine architecture',  'Gothic architecture',  'Medieval Architecture',
                                 'Norman Architecture',  'Romanesque architecture',  'Venetian Gothic Architecture'])

eclecticism = array(['American Foursquare architecture',
                    'American craftsman style',  'Art Nouveau architecture'])

modernism = array(['Art Deco architecture',  'Bauhaus architecture',
                  'Brutalism',  'Chicago school architecture',  'International style'])

neoclassicism = array(['Beaux-Arts architecture',
                      'Greek Revival architecture',  'Palladian architecture'])

renaissance_and_colonialism = array(['Colonial architecture', 'Georgian architecture',   'Northern Renaissance Architecture',  'Spanish Renaissance Architecture',
                                     'French Renaissance Architecture',  'Mannerist Architecture',  'Spanish Colonial Architecture'])

revivalism = array(['Colonial Revival archtecture',  'Orientalism Architecture',  'Russian Revival architecture',
                    'Edwardian architecture', 'Queen Anne architecture',   'Tudor Revival architecture'])

class_predictions = array(['Achaemenid architecture',
                           'Ancient Egyptian architecture',
                           'Andean Baroque Architecture',
                           'Art Deco architecture',
                           'Baroque architecture',
                           'Bauhaus architecture',
                           'Beaux-Arts architecture',
                           'Blobitecture',
                           'Brutalism',
                           'Byzantine architecture',
                           'Chicago school architecture',
                           'Colonial Revival archtecture',
                           'Colonial architecture',
                           'Deconstructivism',
                           'Earthquake Baroque Architecture',
                           'Eco-architecture',
                           'Edwardian architecture',
                           'French Renaissance Architecture',
                           'Georgian architecture',
                           'Gothic architecture',
                           'Greek Revival architecture',
                           'Herodian architecture',
                           'International style',
                           'Mannerist Architecture',
                           'Medieval Architecture',
                           'Neo-futurism architecture',
                           'Norman Architecture',
                           'Northern Renaissance Architecture',
                           'Orientalism Architecture',
                           'Palladian architecture',
                           'Postmodern architecture',
                           'Queen Anne architecture',
                           'Rococo Architecture',
                           'Roman Classical architecture',
                           'Romanesque architecture',
                           'Russian Baroque Architecture',
                           'Russian Revival architecture',
                           'Sicilian Baroque Architecture',
                           'Spanish Colonial Architecture',
                           'Spanish Renaissance Architecture',
                           'Tudor Revival architecture',
                           'Venetian Gothic Architecture'])


def what_epoch(style):
    if style in twety_first_century:
        return '21th Centery'
    elif style in baroque:
        return  'Baroque'
    elif style in classical:
        return 'Classical'
    elif style in early_christian_medieval:
        return 'Early Christian Medievial'
    elif style in eclecticism:
        return 'Ectecticism'
    elif style in modernism:
        return 'Modernism'
    elif style in neoclassicism: 
        return 'Neoclassicism'
    elif style in renaissance_and_colonialism:
        return "Renaissance an Colonialism"
    elif style in revivalism:
        return "Revialism"
    else:
        return 'It is imposible!'

def predict(image_name, image: Image.Image):
    global model
    if model is None:
        model = load_model()

    image = np.asarray(image.resize((512, 512)))[..., :3]

    pred = model.predict(image[None, ...])
    score = softmax(pred[0])

    class_prediction = class_predictions[argmax(score)]
    model_score = round(max(score) * 100, 2)

    period = what_epoch(class_prediction)

    return {
        "image": image_name,
        "model-prediction": class_prediction,
        "model-prediction-period": period,
        "model-prediction-confidence-score": model_score
    }


def read_imagefile(file) -> Image.Image:
    image = Image.open(BytesIO(file))
    return image


@ app.get("/")
async def root():
    return {"message": "Welcome to the Architecture style API!"}


@ app.post("/net/image/prediction/")
async def get_net_image_prediction(image_link: str = ""):
    if image_link == "":
        return {"message": "No image link provided"}

    img_path = get_file(
        origin=image_link
    )
    img = load_img(
        img_path,
        target_size=(224, 224)
    )

    img_array = img_to_array(img)
    img_array = expand_dims(img_array, 0)

    pred = model.predict(img_array)
    score = softmax(pred[0])

    class_prediction = class_predictions[argmax(score)]
    model_score = round(max(score) * 100, 2)

    return {
        "model-prediction": class_prediction,
        "model-prediction-confidence-score": model_score
    }

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    run(app, host="0.0.0.0", port=port)


@ app.post("/upload")
async def predict_api(file: UploadFile = File(...)):
    extension = file.filename.split(".")[-1] in ("jpg", "jpeg", "png")
    if not extension:
        return "Image must be jpg or png format!"
    image = read_imagefile(await file.read())
    prediction = predict(file.filename, image)
    return prediction
