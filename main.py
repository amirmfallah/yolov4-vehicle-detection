import tkinter
import tkinter.messagebox
import customtkinter
import cv2
import PIL
import sys
import time
import os
from datetime import datetime
import numpy as np

CONF_THRESH, NMS_THRESH = 0.4, 0.7

from threading import Thread
if sys.version_info[0] == 2:
    import Queue as queue
else:
    import queue



customtkinter.set_appearance_mode("Dark")  # Modes: "System" (standard), "Dark", "Light"
customtkinter.set_default_color_theme("blue")  # Themes: "blue" (standard), "green", "dark-blue"
global cap

net = cv2.dnn.readNet(os.path.join(os.path.dirname(os.path.realpath(__file__)), "model", "yolov3_custom_last.weights"),os.path.join(os.path.dirname(os.path.realpath(__file__)), "model", "yolov3_custom.cfg"))
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
layers = net.getLayerNames()
output_layers = [layers[i - 1] for i in net.getUnconnectedOutLayers()]

def returnCameraIndexes():
    # checks the first 10 indexes.
    index = 0
    arr = []
    i = 10
    while i > 0:
        cap = cv2.VideoCapture(index)

        #cap = cv2.VideoCapture(os.path.join(os.path.dirname(os.path.realpath(__file__)), "model", "test2.mp4"))
        if cap.read()[0]:
            arr.append(str(index))
            cap.release()
        index += 1
        i -= 1
    return arr

class App(customtkinter.CTk):
    def __init__(self, cams):
        super().__init__()
        self.cap = None
        self.classes = []
        self.notify_name = "Ambulance"
        classesFile = os.path.join(os.path.dirname(os.path.realpath(__file__)), "model", "classes.names")
        with open(classesFile, 'r') as f:
            self.classes = [line.strip() for line in f.readlines()]

        # configure window
        self.title("Emergency Vehicle Detection YOLO")
        self.geometry(f"{1100}x{580}")

        # configure grid layout (4x4)
        self.grid_columnconfigure(1, weight=1)

        # create sidebar frame with widgets
        self.sidebar_frame = customtkinter.CTkFrame(self, width=140, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, rowspan=4, sticky="nsew")
        self.logo_label = customtkinter.CTkLabel(self.sidebar_frame, text="Emergency Vehicle Detection", font=customtkinter.CTkFont(size=20, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=10)


        self.start_button = customtkinter.CTkButton(self.sidebar_frame, command=self.start, text="start")
        self.start_button.grid(row=8, column=0, padx=20, pady=10)

        self.cameras = customtkinter.CTkLabel(self.sidebar_frame, text="Available cameras:", anchor="w")
        self.cameras.grid(row=3, column=0, padx=20, pady=10)

        self.cameras_options = customtkinter.CTkOptionMenu(self.sidebar_frame, values=cams,
                                                               command=self.change_camera)
        self.cameras_options.grid(row=5, column=0, padx=20, pady=10)


        self.notifyL = customtkinter.CTkLabel(self.sidebar_frame, text="Notification Class:", anchor="w")
        self.notifyL.grid(row=6, column=0, padx=20, pady=10)
        self.notifyL_options = customtkinter.CTkOptionMenu(self.sidebar_frame, values=self.classes,
                                                               command=self.notify)
        self.notifyL_options.grid(row=7, column=0, padx=20, pady=10)

        # create textbox
        self.textbox = customtkinter.CTkTextbox(self, width=210)
        self.textbox.grid(row=1, column=1, padx=(10, 10), pady=(10, 10), sticky="nsew")

        self.main_frame = customtkinter.CTkFrame(self, width=1, corner_radius=5)
        self.main_frame.grid(row=0, column=1)

        image_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "assets")
        self.frame = customtkinter.CTkImage(PIL.Image.open(os.path.join(image_path, "placeholder.jpg")), size=(416, 312))
        self.image_label = customtkinter.CTkLabel(self.main_frame, image=self.frame, text="")  # display image with a CTkLabel
        self.image_label.grid(row=0, column=1, padx=20, pady=20)


    def change_camera(self, new):
        self.selected_camera = new

    def start(self):
        try:
          self.cap = cv2.VideoCapture(int(self.selected_camera))
          #self.cap = cv2.VideoCapture('assets/test.mp4')
        except:
          self.textbox.insert("0.0", str(datetime.now()) + ": Error while opening camera\n")

        self.update()

    def update(self):
      ret, frame = self.cap.read()
      if ret == False:
          return

      height, width = frame.shape[:2]
      blob = cv2.dnn.blobFromImage(frame, 1/256, (480, 480), swapRB=True, crop=False)
      net.setInput(blob)
      layer_outputs = net.forward(output_layers)
      class_ids, confidences, b_boxes = [], [], []
      for output in layer_outputs:
          for detection in output:
              scores = detection[5:]
              class_id = np.argmax(scores)
              confidence = scores[class_id]

              if confidence > CONF_THRESH:
                  center_x, center_y, w, h = (detection[0:4] * np.array([width, height, width, height])).astype('int')

                  x = int(center_x - w / 2)
                  y = int(center_y - h / 2)

                  b_boxes.append([x, y, int(w), int(h)])
                  confidences.append(float(confidence))
                  class_ids.append(int(class_id))

      if len(b_boxes):
          indices = cv2.dnn.NMSBoxes(b_boxes, confidences, CONF_THRESH, NMS_THRESH).flatten().tolist()

          classes = os.path.join(os.path.dirname(os.path.realpath(__file__)), "model", "classes.names")
          with open(classes, 'r') as f:
              classes = [line.strip() for line in f.readlines()]
          colors = np.random.uniform(0, 255, size=(len(classes), 3))

          for index in indices:
              x, y, w, h = b_boxes[index]
              cv2.rectangle(frame, (x, y), (x + w, y + h), colors[index], 2)
              cv2.putText(frame, classes[class_ids[index]], (x + 5, y + 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, colors[index],2)

      frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
      img = PIL.Image.fromarray(frame)
      self.frame.configure(light_image=img, dark_image=img)
      self.main_frame.after(1, self.update)


    def get_cap(self):
        return self.cap

    def notify(self, new):
        self.notify_name = new
        print(new)

class QueueFPS(queue.Queue):
    def __init__(self):
        queue.Queue.__init__(self)
        self.startTime = 0
        self.counter = 0

    def put(self, v):
        queue.Queue.put(self, v)
        self.counter += 1
        if self.counter == 1:
            self.startTime = time.time()

    def getFPS(self):
        return self.counter / (time.time() - self.startTime)

framesQueue = QueueFPS()
def framesThreadBody():
    global framesQueue, process

    while process:
        hasFrame, frame = cap.read()
        if not hasFrame:
            break
        framesQueue.put(frame)

if __name__ == "__main__":
    #cams = returnCameraIndexes()
    cams = ["0", "1", "2", "3", "4"]
    app = App(cams)
    cap = app.get_cap()
    app.mainloop()
