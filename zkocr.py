import easyocr
import cv2
import numpy as np
import json
import re
from PIL import Image
from datetime import datetime

# Load the image
image_path = './ine.png'
img = Image.open(image_path)

# Convert image to grayscale
img = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2GRAY)

# Apply thresholding to preprocess the image
_, img = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Save the preprocessed image for inspection
cv2.imwrite('preprocessed_ine.png', img)

# Initialize the EasyOCR reader
reader = easyocr.Reader(['es'])  # Use 'es' for Spanish

# Perform OCR on the image
result = reader.readtext(img, detail=0)

# Print the OCR result
print("OCR Result:")
print(result)

# Join the extracted text into a single string
extracted_text = "\n".join(result)
print("Extracted Text:")
print(extracted_text)

# Split the text into lines for better parsing
lines = extracted_text.split("\n")

# Extract the birth date
birthdate_pattern = re.compile(r'\d{2}/\d{2}/\d{4}')
birthdate = None
for line in lines:
    if 'FECHA DE NACIMIENTO' in line:
        for i in range(len(lines)):
            if birthdate_pattern.search(lines[i]):
                birthdate = birthdate_pattern.search(lines[i]).group(0)
                break

# Calculate the age
def calculate_age(birthdate_str):
    birthdate = datetime.strptime(birthdate_str, "%d/%m/%Y")
    today = datetime.today()
    age = today.year - birthdate.year - ((today.month, today.day) < (birthdate.month, birthdate.day))
    return age

age = calculate_age(birthdate) if birthdate else None

# Create the JSON object
data = {
    "birthDate": birthdate,
    "age": age
}

# Print the JSON object
print("JSON Output:")
print(json.dumps(data, indent=4, ensure_ascii=False))
