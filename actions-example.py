from giza_actions.action import Action, action
from giza_actions.task import task
import easyocr
import cv2
import numpy as np
from PIL import Image
from datetime import datetime
import onnxruntime as ort
import json
import re
from giza_actions.model import GizaModel

@task
def preprocess_image(image_path):
    img = Image.open(image_path)
    img = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2GRAY)
    _, img = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return img

@task
def perform_ocr(img):
    reader = easyocr.Reader(['es'])
    result = reader.readtext(img, detail=0)
    return result

@task
def extract_birthdate(result):
    extracted_text = "\n".join(result)
    lines = extracted_text.split("\n")
    birthdate_pattern = re.compile(r'\d{2}/\d{2}/\d{4}')
    birthdate = None
    for line in lines:
        if 'FECHA DE NACIMIENTO' in line:
            for i in range(len(lines)):
                if birthdate_pattern.search(lines[i]):
                    birthdate = birthdate_pattern.search(lines[i]).group(0)
                    break
    return birthdate

@task
def calculate_age_with_onnx(birthdate):
    def parse_birthdate(birthdate_str):
        birthdate = datetime.strptime(birthdate_str, "%d/%m/%Y")
        return [birthdate.year, birthdate.month, birthdate.day]

    birthdate_parsed = parse_birthdate(birthdate)
    model_path = "./age_calculator_model.onnx"
    session = ort.InferenceSession(model_path)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    input_feed = {input_name: np.array([birthdate_parsed], dtype=np.float32)}
    result = session.run([output_name], input_feed)
    age = int(round(result[0][0].item()))
    return age

@task
def verifiable_inference(image, model_id, version_id):
    model = GizaModel(id=model_id, version=version_id)
    result, request_id = model.predict(input_feed={"image": image}, verifiable=True)
    return result, request_id

@action
def process_image(image_path):
    img = preprocess_image(image_path)
    result = perform_ocr(img)
    birthdate = extract_birthdate(result)
    age = calculate_age_with_onnx(birthdate)
    data = {
        "birthDate": birthdate,
        "age": age
    }
    print("JSON Output:")
    print(json.dumps(data, indent=4, ensure_ascii=False))

@action
def process_image_verifiable(image_path):
    img = preprocess_image(image_path)
    result = perform_ocr(img)
    birthdate = extract_birthdate(result)
    # Assuming the image is used for the verifiable inference
    verifiable_result, request_id = verifiable_inference(img, 1, 1)  # Replace 1, 1 with actual model_id and version_id
    data = {
        "birthDate": birthdate,
        "verifiable_result": verifiable_result,
        "request_id": request_id
    }
    print("JSON Output:")
    print(json.dumps(data, indent=4, ensure_ascii=False))

if __name__ == '__main__':
    # Deploying the unverifiable action
    action_deploy_unverifiable = Action(entrypoint=process_image, name="process-image-action")
    action_deploy_unverifiable.serve(name="process-image-action-deployment")

    # Deploying the verifiable action
    action_deploy_verifiable = Action(entrypoint=process_image_verifiable, name="process-image-verifiable-action")
    action_deploy_verifiable.serve(name="process-image-verifiable-deployment")
