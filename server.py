import argparse
import os
import cv2
import imutils
from predict import predict

class App():
    def __init__(self, video_path):
        self.camera = cv2.VideoCapture(video_path)
        self.detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    def __del__(self):
        print("delect")
    
    def run(self):
        while True:
            (grabbed, frame) = self.camera.read()
            if not grabbed:
                break
            frame = imutils.resize(frame, width=480)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frameClone = frame.copy()
            rects = self.detector.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, flags=cv2.CASCADE_SCALE_IMAGE)
            for (fX, fY, fW, fH) in rects:
                roi = gray[fY:fY + fH, fX:fX + fW]
                roi = roi.astype('float32')
                label = predict(roi)
                # label = 'Smile' if label == 0 else 'Not smile'
                if label == 0:
                    label = 'Smile'
                    color = (255, 0, 0)
                else:
                    label = 'Not smile'
                    color = (0, 0, 255)
                cv2.putText(frameClone, label, (fX, fY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
                cv2.rectangle(frameClone, (fX, fY), (fX + fW, fY + fH), color, 2)
            
            cv2.imshow("Face", frameClone)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        self.camera.release()
        cv2.destroyAllWindows()
  
parser = argparse.ArgumentParser(description='Emotion detection, type "q" for exit')
parser.add_argument('-v', '--video', default='', help='Mp4 video path (Optional')
args = parser.parse_args()

if __name__ == '__main__':
    if os.path.exists(args.video):
        mode = args.video
    else :
        mode = 0
        print(f"Can not open file {args.video}, opening webcam instead")
    app = App(mode)
    app.run()
    del app 