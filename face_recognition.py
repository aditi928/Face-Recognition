
import io
import os
import cv2
import base64
import numpy as np
import PIL.Image as Image

import _face_detection as ftk

face_dir = "faces/"


class FaceDetection:
    verification_threshold = 0.8
    v, net = None, None
    image_size = 160
    embeddings = {}

    def __init__(self):
        FaceDetection.load_models()

    @staticmethod
    def load_models():
        if not FaceDetection.net:
            FaceDetection.net = FaceDetection.load_opencv()

        if not FaceDetection.v:
            FaceDetection.v = FaceDetection.load_face_detection()
        
    @staticmethod
    def load_opencv():
        model_path = "./Models/OpenCV/opencv_face_detector_uint8.pb"
        model_pbtxt = "./Models/OpenCV/opencv_face_detector.pbtxt"
        net = cv2.dnn.readNetFromTensorflow(model_path, model_pbtxt)
        return net

    @staticmethod
    def load_face_detection():
        v = ftk.Verification()
        v.load_model("./Models/FaceDetection/")
        v.initial_input_output_tensors()
        return v

    # Convert base64 to image
    @staticmethod
    def base64_to_numpy(base64_img):
        img_data = bytes(base64_img, encoding='utf-8')
        with open("test.png", "wb") as fh:
            fh.write(base64.decodebytes(img_data))

        img = cv2.imread("test.png")
        try:
            os.remove("test.png")
        except PermissionError:
            pass
        return img

    # Convert image to base64
    @staticmethod
    def get_response_image(image):
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img)
        byte_arr = io.BytesIO()
        pil_img.save(byte_arr, format='PNG')  # convert the PIL image to byte array
        encoded_img = base64.encodebytes(byte_arr.getvalue()).decode('ascii')  # encode as base64
        return encoded_img

    @staticmethod
    def is_same(emb1, emb2):
        diff = np.subtract(emb1, emb2)
        diff = np.sum(np.square(diff))
        return diff < FaceDetection.verification_threshold, diff

    @staticmethod
    def detect_faces(image, display_images=False):  # Make display_image to True to manually debug errors
        height, width, channels = image.shape

        blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), [104, 117, 123], False, False)
        FaceDetection.net.setInput(blob)
        detections = FaceDetection.net.forward()

        faces = []

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:
                x1 = int(detections[0, 0, i, 3] * width)
                y1 = int(detections[0, 0, i, 4] * height)
                x2 = int(detections[0, 0, i, 5] * width)
                y2 = int(detections[0, 0, i, 6] * height)
                faces.append([x1, y1, x2 - x1, y2 - y1])

                if display_images:
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 3)
    
        if display_images:
            print("Face co-ordinates: ", faces)
            cv2.imshow("Training Face", cv2.resize(image, (300, 300)))
            cv2.waitKey(0)
        return faces

    @staticmethod
    def load_face_embeddings(image_dir, force=False):
        if FaceDetection.embeddings != {} and not force:
            return FaceDetection.embeddings

        embeddings = {}
        for file in os.listdir(image_dir):
            img_path = image_dir + file
            try:
                image = cv2.imread(img_path)
                faces = FaceDetection.detect_faces(image)
                if len(faces) == 1:
                    x, y, w, h = faces[0]
                    image = image[y:y + h, x:x + w]
                    embeddings[file.split(".")[0]] = FaceDetection.v.img_to_encoding(cv2.resize(image, (160, 160)), FaceDetection.image_size)
                else:
                    print(f"Found more than 1 face in \"{file}\", skipping embeddings for the image.")
            except Exception:
                print(f"Unable to read file: {file}")

        FaceDetection.embeddings = embeddings
        return embeddings

    @staticmethod
    def fetch_detections(image, embeddings):
        
        faces = FaceDetection.detect_faces(image)
        
        detections = []
        for face in faces:
            x, y, w, h = face
            im_face = image[y:y + h, x:x + w]
            img = cv2.resize(im_face, (200, 200))
            user_embed = FaceDetection.v.img_to_encoding(cv2.resize(img, (160, 160)), FaceDetection.image_size)
            
            detected = {}
            for _user in embeddings:
                flag, thresh = FaceDetection.is_same(embeddings[_user], user_embed)
                if flag:
                    detected[_user] = thresh
            
            detected = {k: v for k, v in sorted(detected.items(), key=lambda item: item[1])}
            detected = list(detected.keys())
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 3)

            if len(detected) > 0:
                detections.append(detected[0])
                cv2.putText(image, detected[0], (x, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        encoded_image = "data:image/png;base64, " + FaceDetection.get_response_image(image)

        return {"data": detections, "image": encoded_image}


def face_recognition_api(image, custom=False):
    FaceDetection.load_models()
    if custom:
        npimg = np.frombuffer(image, np.uint8)
        image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    else:
        image = FaceDetection.base64_to_numpy(image)
    embeddings = FaceDetection.load_face_embeddings(face_dir)
    response = FaceDetection.fetch_detections(image, embeddings)
    return response


def get_face_list():
    return list(FaceDetection.embeddings.keys())


def force_reload_embeddings():
    FaceDetection.load_face_embeddings(face_dir, force=True)


def initialize_test():
    FaceDetection.load_models()
    embeddings = FaceDetection.load_face_embeddings(face_dir)
    FaceDetection.fetch_detections(cv2.imread('static/test_image/test.jpg'), embeddings)
