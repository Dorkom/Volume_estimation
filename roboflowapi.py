# import the necessary libraries
from inference_sdk import InferenceHTTPClient
import json

# initialize the client
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="KOYZaBjvJzk174XAygWU"
)

# infer on a local image
result = CLIENT.infer('C:/Users/varer/Documents/Proyectogallos/MICAI_FINALE/frames_de_prueba_sin_calibrar/frame_000031.jpg', model_id="etiquetado-bloques-keypoints/3")

# save the result to a JSON file
with open("resultado.json", "w") as json_file:
    json.dump(result, json_file)

print("Resultado guardado en resultado.json")
