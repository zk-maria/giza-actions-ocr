### Project Name: zk-maria

### Tagline:
Securely verify age for government permissions without revealing identity.

### The problem it solves:
Ensures individuals meet age requirements for government permissions while preserving their privacy by not revealing personal information.

### Challenges we ran into:
- Integrating OCR for accurate text extraction from IDs.
- Training and deploying ONNX models for age calculation.
- Implementing verifiable ML models using GizaModel for secure, provable age verification.

### Technologies we used:
- **Python**: Programming language for developing the project.
- **EasyOCR**: Library for extracting text from images.
- **OpenCV**: Image processing library.
- **ONNX Runtime**: For running machine learning models.
- **GizaModel**: For verifiable ML inference using Zero-Knowledge proofs.
- **Giza Actions**: For task and action orchestration.
- **PIL (Pillow)**: Python Imaging Library for image manipulation.

### How it works:
1. **Image Preprocessing**: Convert the ID image to grayscale and apply thresholding to prepare it for OCR.
2. **OCR Text Extraction**: Use EasyOCR to extract text from the preprocessed image.
3. **Birthdate Extraction**: Identify and extract the birthdate from the OCR results.
4. **Age Calculation**: Use an ONNX model to calculate the age based on the extracted birthdate.
5. **Verifiable Inference**: For secure verification, use GizaModel to perform verifiable inference, providing a proof that the age calculation was correct.
6. **Privacy Preservation**: Only the calculated age and proof of calculation are shared, ensuring the individual's identity remains private.

### Diagram (image):

Here is a simple diagram to illustrate the workflow:

```plaintext
+------------------+        +------------------+        +------------------+        +------------------+
|                  |        |                  |        |                  |        |                  |
|   Input ID Image +------->+   Preprocess     +------->+  OCR Text        +------->+  Extract Birthdate|
|                  |        |   Image          |        |  Extraction      |        |                  |
|                  |        |                  |        |                  |        |                  |
+------------------+        +------------------+        +------------------+        +--------+---------+
                                                                                             |
                                                                                             v
                                                                                   +---------+---------+
                                                                                   |                   |
                                                                                   |   Age Calculation |
                                                                                   |   with ONNX       |
                                                                                   |                   |
                                                                                   +---------+---------+
                                                                                             |
                                                                                             v
                                                                                   +---------+---------+
                                                                                   |                   |
                                                                                   |   Verifiable      |
                                                                                   |   Inference with  |
                                                                                   |   GizaModel       |
                                                                                   |                   |
                                                                                   +---------+---------+
                                                                                             |
                                                                                             v
                                                                                   +---------+---------+
                                                                                   |                   |
                                                                                   |  Output Age and   |
                                                                                   |  Proof            |
                                                                                   |                   |
                                                                                   +-------------------+
```

This diagram shows the step-by-step process from inputting the ID image to producing the age and proof of calculation without revealing personal identity details.
