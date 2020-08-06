import os
import cv2
import pickle
import msvcrt
import numpy as np
import face_recognition

class ImageRecognition(object):
    def __init__(self):
        self.image_folder_path = "Images"
        self.resize_factor = 2
        self.known_images_encoded_data = []
        self.names = []
        self.images_file_names = os.listdir(self.image_folder_path)
        self.set_image_data()

    def get_image_encoded_data(self, image_object):
            image_object_RGB = cv2.cvtColor(image_object, cv2.COLOR_BGR2RGB)
            encoded_data = face_recognition.face_encodings(image_object_RGB)[0]
            return encoded_data

    def load_data_from_file(self):
            try:
                with open("image_data.pkl", "rb") as file_object:
                    data_dict = pickle.load(file_object)

                return data_dict
            except:
                return {}

    def save_face_encode_data(self):
            data_dict = dict(zip(self.names, self.known_images_encoded_data))
            with open("image_data.pkl", "wb") as file_object:
                    pickle.dump(data_dict, file_object)

    def set_image_data(self):
        data_dict_from_file = self.load_data_from_file()
        for image_filepath in self.images_file_names:
            name = os.path.splitext(image_filepath)[0]
            self.names.append(name)
            if name in data_dict_from_file.keys():
                self.known_images_encoded_data.append(data_dict_from_file[name])
                continue

            image_object = cv2.imread(f'{self.image_folder_path}/{image_filepath}')
            self.known_images_encoded_data.append(self.get_image_encoded_data(image_object))

        self.save_face_encode_data()

    def resize_and_convert_to_RGB(self, image):
        resized_image = cv2.resize(
            image,
            (0,0),
            None,
            (1/self.resize_factor),
            (1/self.resize_factor)
        )
        return cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)

    def show_match(self, face_location, index, image):
        name = self.names[index]
        y1, x2, y2, x1 = face_location
        y1, x1, y2, x2 = y1 * self.resize_factor, x1 * self.resize_factor, y2 * self.resize_factor, x2 * self.resize_factor
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 1)
        cv2.rectangle(image, (x1, y2 - 23), (x2, y2), (0, 255, 0), cv2.FILLED)
        cv2.putText(image,name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

    def recognise_faces(self, face_locations, encoded_data, image):
        for face_location, encoded_data in zip(face_locations, encoded_data):
            matches = face_recognition.compare_faces(self.known_images_encoded_data, encoded_data)
            face_distance = face_recognition.face_distance(self.known_images_encoded_data, encoded_data)
            match_index = np.argmin(face_distance)
            if matches[match_index]:
                self.show_match(face_location, match_index, image)

    def start_recognition_from_web_cam(self):
        print("Switching on WebCam")
        capture = cv2.VideoCapture(0)
        flag = True
        print("Press 'q' to quit")
        cv2.namedWindow("Web Cam", cv2.WINDOW_NORMAL)
        # cv2.setWindowProperty("Web Cam", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        while flag:
            sucess, cam_image_orginal = capture.read()
            cam_image = self.resize_and_convert_to_RGB(cam_image_orginal)

            face_locations_current_frame = face_recognition.face_locations(cam_image)
            encoded_data_current_frame= face_recognition.face_encodings(cam_image, face_locations_current_frame)

            self.recognise_faces(face_locations_current_frame, encoded_data_current_frame, cam_image_orginal)

            cv2.imshow("Web Cam", cam_image_orginal)
            cv2.waitKey(1)

            if msvcrt.kbhit():
                key = str(msvcrt.getch())
                if key == "b'q'":
                    flag = False
                    print("Quiting")

if __name__ == "__main__":
    recognition = ImageRecognition()
    recognition.start_recognition_from_web_cam()
