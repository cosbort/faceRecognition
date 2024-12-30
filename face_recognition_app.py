import cv2
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import threading
import time

class FaceRecognitionApp:
    def __init__(self, window):
        self.window = window
        self.window.title("Riconoscimento Facciale")
        self.window.configure(bg='#2c3e50')
        
        # Stile dell'applicazione
        style = ttk.Style()
        style.theme_use('default')
        style.configure('Custom.TButton', 
                       background='#3498db',
                       foreground='black',
                       padding=10,
                       font=('Helvetica', 12, 'bold'))
        style.map('Custom.TButton',
                 foreground=[('active', 'white')],
                 background=[('active', '#2980b9')])
        
        # Frame principale
        self.main_frame = ttk.Frame(window)
        self.main_frame.pack(padx=20, pady=20)
        
        # Area video
        self.video_label = tk.Label(self.main_frame)
        self.video_label.pack()
        
        # Pulsanti
        self.button_frame = ttk.Frame(self.main_frame)
        self.button_frame.pack(pady=20)
        
        self.start_button = ttk.Button(self.button_frame, 
                                     text="Avvia Camera",
                                     command=self.start_video,
                                     style='Custom.TButton')
        self.start_button.pack(side=tk.LEFT, padx=5)
        
        self.stop_button = ttk.Button(self.button_frame,
                                    text="Stop",
                                    command=self.stop_video,
                                    style='Custom.TButton')
        self.stop_button.pack(side=tk.LEFT, padx=5)
        
        # Stato iniziale
        self.is_running = False
        self.video_capture = None
        
        # Carica il classificatore per il riconoscimento facciale
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_eye.xml'
        )
    
    def start_video(self):
        if not self.is_running:
            self.video_capture = cv2.VideoCapture(0)
            self.is_running = True
            self.video_thread = threading.Thread(target=self.update_frame)
            self.video_thread.daemon = True
            self.video_thread.start()
    
    def stop_video(self):
        self.is_running = False
        if self.video_capture is not None:
            self.video_capture.release()
    
    def update_frame(self):
        while self.is_running:
            ret, frame = self.video_capture.read()
            if ret:
                # Converti in scala di grigi per il rilevamento
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Rileva i volti
                faces = self.face_cascade.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
                )
                
                # Disegna rettangoli intorno ai volti
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                    
                    # Regione di interesse per gli occhi
                    roi_gray = gray[y:y+h, x:x+w]
                    roi_color = frame[y:y+h, x:x+w]
                    
                    # Rileva gli occhi
                    eyes = self.eye_cascade.detectMultiScale(roi_gray)
                    for (ex, ey, ew, eh) in eyes:
                        cv2.rectangle(roi_color, (ex, ey), 
                                    (ex+ew, ey+eh), (0, 255, 0), 2)
                
                # Converti l'immagine per tkinter
                cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(cv2image)
                imgtk = ImageTk.PhotoImage(image=img)
                
                self.video_label.imgtk = imgtk
                self.video_label.configure(image=imgtk)
            
            time.sleep(0.03)  # Limita il framerate

    def __del__(self):
        if self.video_capture is not None:
            self.video_capture.release()

if __name__ == "__main__":
    root = tk.Tk()
    app = FaceRecognitionApp(root)
    root.mainloop()
