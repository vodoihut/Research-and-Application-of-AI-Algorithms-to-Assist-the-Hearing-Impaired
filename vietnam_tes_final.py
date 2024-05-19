# Importing Libraries
import numpy as np
import math
import unicodedata
import cv2
import os, sys
import traceback
import pyttsx3
import speech_recognition as sr
from keras.models import load_model
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
from string import ascii_uppercase
import enchant
ddd=enchant.Dict("vi_VN")
hd = HandDetector(maxHands=1)
hd2 = HandDetector(maxHands=1)
import tkinter as tk
from PIL import Image, ImageTk

offset=29
img_size = 400
# labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L',
#           'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W','X', 'Y', 'Z']
labels = ['A', 'B', 'C', 'D', 'E', 'G', 'H', 'I', 'K', 'L',
          'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W','X', 'Y', 'dấu cách',
          'tiếp tục', 'dấu mũ', 'dấu râu']
# os.environ["THEANO_FLAGS"] = "device=cuda, assert_no_cpu_op=True"
os.environ["THEANO_FLAGS"] = "device=cpu, assert_no_cpu_op=True"

def check_sim(i, file_map):
    for item in file_map:
        for word in file_map[item]:
            if (i == word):
                return 1, item
    return -1, ""




alpha_dest = "C:\\Users\\hieu\\PycharmProjects\\handsign\\two-way-sign-language-translator-master\\alphabet\\"
output_dir = "C:\\Users\\hieu\\PycharmProjects\\handsign\\hand_new\\save"

file_map = {}
char_dict = {'à': 'a ' + 'dauhuyen',
             'á': 'a ' + 'dausac',
             'ả': 'a ' + 'dauhoi',
             'ạ': 'a ' + 'daunang',
             'ã': 'a ' + 'daunga',
             'ă': 'a ' + 'daurau',
             'ắ': 'a ' + 'daurau ' + 'dausac',
            'ằ': 'a ' + 'daurau ' + 'dauhuyen',
            'ẳ': 'a ' + 'daurau ' + 'dauhoi',
            'ẵ': 'a ' + 'daurau ' + 'daunga',
            'ặ': 'a ' + 'daurau ' + 'daunang',
             'â': 'a ' + 'daumu',
            'ấ': 'a ' + 'daumu ' + 'dausac',
            'ầ': 'a ' + 'daumu ' + 'dauhuyen',
            'ẩ': 'a ' + 'daumu ' + 'dauhoi',
            'ẫ': 'a ' + 'daumu ' + 'daunga',
            'ậ': 'a ' + 'daumu ' + 'daunang',
             'é': 'e ' + 'dausac',
            'è': 'e ' + 'dauhuyen',
            'ẻ': 'e ' + 'dauhoi',
            'ẹ': 'e ' + 'daunang',
            'ẽ': 'e ' + 'daunga',
             'ê': 'e ' + 'daumu',
            'ế': 'e ' + 'daumu ' + 'dausac',
            'ề': 'e ' + 'daumu ' + 'dauhuyen',
            'ể': 'e ' + 'daumu ' + 'dauhoi',
            'ệ': 'e ' + 'daumu ' + 'daunang',
            'ễ': 'e ' + 'daumu ' + 'daunga',
            'ì': 'i ' + 'dauhuyen',
            'í': 'i ' + 'dausac',
            'ỉ': 'i ' + 'dauhoi',
            'ị': 'i ' + 'daunang',
            'ĩ': 'i ' + 'daunga',
            'ò': 'o ' + 'dauhuyen',
            'ó': 'o ' + 'dausac',
            'ỏ': 'o ' + 'dauhoi',
            'ọ': 'o ' + 'daunang',
            'õ': 'o ' + 'daunga',
            'ô': 'o ' + 'daumu',
            'ồ': 'o ' + 'daumu ' + 'dauhuyen',
            'ố': 'o ' + 'daumu ' + 'dausac',
            'ổ': 'o ' + 'daumu ' + 'dauhoi',
            'ộ': 'o ' + 'daumu ' + 'daunang',
            'ỗ': 'o ' + 'daumu ' + 'daunga',
            'ơ': 'o ' + 'daurau',
            'ờ': 'o ' + 'daurau ' + 'dauhuyen',
            'ớ': 'o ' + 'daurau ' + 'dausac',
            'ở': 'o ' + 'daurau ' + 'dauhoi',
            'ợ': 'o ' + 'daurau ' + 'daunang',
            'ỡ': 'o ' + 'daurau ' + 'daunga',
            'ù': 'u ' + 'dauhuyen',
            'ú': 'u ' + 'dausac',
            'ủ': 'u ' + 'dauhoi',
            'ụ': 'u ' + 'daunang',
            'ũ': 'u ' + 'daunga',
            'ư': 'u ' + 'daurau',
            'ừ': 'u ' + 'daurau ' + 'dauhuyen',
            'ứ': 'u ' + 'daurau ' + 'dausac',
            'ử': 'u ' + 'daurau ' + 'dauhoi',
            'ự': 'u ' + 'daurau ' + 'daunang',
            'ữ': 'u ' + 'daurau ' + 'daunga',
            'ỳ': 'y' + 'dauhuyen',
            'ý': 'y' + 'dausac',
            'ỷ': 'y' + 'dauhoi',
            'ỵ': 'y' + 'daunang',
            'ỹ': 'y' + 'daunga',
             }

def process_char(char):
    if char in char_dict:
        return char_dict[char].split(' ')
    return char

def func(text):
    all_frames = []
    final = Image.new('RGB', (380, 260))
    words = text.split()
    for word in words:
        for char in word:
            processed_chars = process_char(char)
            for processed_char in processed_chars:
                print(processed_char)
                im_name = f"{processed_char.lower()}_small.gif"
                im_path = os.path.join(alpha_dest, im_name)
                im = Image.open(im_path)
                frameCnt = im.n_frames
                for frame_cnt in range(frameCnt):
                    im.seek(frame_cnt)
                    im.save(os.path.join(output_dir, f"{processed_char}_{frame_cnt}.png"))
                    img = cv2.imread(os.path.join(output_dir, f"{processed_char}_{frame_cnt}.png"))
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, (380, 260))
                    im_arr = Image.fromarray(img)
                    for itr in range(15):
                        all_frames.append(im_arr)
    final.save(os.path.join(output_dir, "out.gif"), save_all=False, append_images=all_frames, duration=100, loop=0)
    return all_frames



class Tk_Manage(tk.Tk):
    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)
        self.title("Chuyển đổi ngôn ngữ ký hiệu")
        container = tk.Frame(self)
        container.pack(side="top", fill="both", expand=True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)
        self.frames = {}
        for F in (StartPage, VtoS, Application):
            frame = F(container, self)
            self.frames[F] = frame
            frame.grid(row=0, column=0, sticky="nsew")
        self.show_frame(StartPage)

    def show_frame(self, cont):
        frame = self.frames[cont]
        frame.tkraise()


class StartPage(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.Title_start = tk.Label(self)
        self.Title_start.place(x=400, y=10)
        self.Title_start.config(text="Ứng dụng dịch ngôn ngữ ký hiệu", font=("Courier", 30, "bold"))

        self.TtoS_button = tk.Button(self)
        self.TtoS_button.place(x=350, y=100)
        self.TtoS_button.config(text="Chuyển văn bản/giọng nói sang ngôn ngữ ký hiệu", font=("Courier", 20),bg="light green", fg="black", command=lambda: controller.show_frame(VtoS))

        self.StoT_button = tk.Button(self)
        self.StoT_button.place(x=350, y=160)
        self.StoT_button.config(text="Chuyển ngôn ngữ ký hiệu sang văn bản/giọng nói", font=("Courier", 20),bg="light blue", fg="black",
                                command=lambda: sign_to_voice())

        load = Image.open("C:\\Users\\hieu\\PycharmProjects\\handsign\\hand_new\\background.png")
        load = load.resize((1100, 570))
        render = ImageTk.PhotoImage(load)
        img = tk.Label(self, image=render)
        img.image = render
        img.place(x=200, y=220)

        def sign_to_voice():
            controller.frames[Application].start_webcam()
            controller.show_frame(Application)


class VtoS(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        self.gif_frames = []
        self.cnt = 0
        self.inputtxt = None

        self.Title_VtoS = tk.Label(self)
        self.Title_VtoS.place(x=250, y=10)
        self.Title_VtoS.config(text="Chuyển văn bản/giọng nói sang ngôn ngữ ký hiệu", font=("Courier", 30, "bold"))

        self.result_gif = tk.Label(self)
        self.result_gif.place(x=1150, y=160)
        self.result_gif.config(text="Kết quả", font=("Courier", 20))

        self.gif_box = tk.Label(self)




        def gif_stream():
            if self.cnt == len(self.gif_frames):
                return
            img = self.gif_frames[self.cnt]
            self.cnt += 1
            img_resized = img.resize((450, 450), Image.Resampling.LANCZOS)
            imgtk = ImageTk.PhotoImage(image=img_resized)
            self.gif_box.imgtk = imgtk
            self.gif_box.configure(image=imgtk)
            self.gif_box.after(50, gif_stream)
        def voice_vip():
            delete_text()
            hear_voice()
        def hear_voice():
            delete_text()
            store = sr.Recognizer()
            with sr.Microphone() as source:
                print("Adjusting noise ")
                store.adjust_for_ambient_noise(source, duration=1)
                print("Recording for 3 seconds")
                audio_input = store.listen(source, timeout=3)
                print("Done recording")
                try:
                    text_output = store.recognize_google(audio_input, language="vi-VN")
                    self.inputtxt.insert(tk.END, text_output)
                    print(text_output)
                except:
                    print("Error Hearing Voice")
                    self.inputtxt.insert(tk.END, '')

        def Take_input():
            INPUT = self.inputtxt.get("1.0", "end-1c")
            print(INPUT)
            self.gif_frames = func(INPUT)
            self.cnt = 0
            gif_stream()
            self.gif_box.place(x=1000, y=220)

        def delete_text():
            self.inputtxt.delete("1.0", tk.END)

            self.gif_box.place_forget()
            self.gif_frames = []
            self.cnt = 0
        def back_to_home():
            delete_text()
            self.controller.show_frame(StartPage)


        self.back = tk.Button(self)
        self.back.place(x=690, y=650)
        self.back.config(text="Trở lại", font=("Courier", 20),
                         command=back_to_home)

        self.delete_but = tk.Button(self)
        self.delete_but.place(x=50, y=550)
        self.delete_but.config(text="Xóa", font=("Courier", 20),bg="red", fg="black",
                         command=delete_text)

        self.enter_text = tk.Label(self)
        self.enter_text.place(x=50, y=160)
        self.enter_text.config(text="Văn bản hoặc:", font=("Courier", 20))

        self.inputtxt = tk.Text(self, height=8, width=30)
        self.inputtxt.config(font=("Courier", 20))

        # button voice
        self.voice_button = tk.Button(self)
        self.voice_button.place(x=300, y=150)
        self.voice_button.config(text="Giọng nói", font=("Courier", 20), bg="light green", fg="black",
                                command=voice_vip)

        self.convert_button = tk.Button(self)
        self.convert_button.place(x=600, y=400)
        self.convert_button.config(height=2, width=20, text="Chuyển đổi",font=("Courier", 20), bg="light yellow", fg="black", command=Take_input)

        self.inputtxt.place(x=50, y=270)


# Application :

class Application(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        # self.vs = cv2.VideoCapture(0)
        self.current_image = None
        self.model = Classifier("C:\\Users\\hieu\\PycharmProjects\\handsign\\hand_new\\models\\bestmodel_0505.h5",
                        "C:\\Users\\hieu\\PycharmProjects\\handsign\\hand_new\\models\\labels.txt")
        self.speak_engine=pyttsx3.init()

        self.speak_engine.setProperty("rate",100)
        voices=self.speak_engine.getProperty("voices")
        self.speak_engine.setProperty("voice",voices[1].id)


        self.ct = {}
        self.ct['blank'] = 0
        self.blank_flag = 0
        self.space_flag=False
        self.next_flag=True
        self.prev_char=""
        self.count=-1
        self.ten_prev_char=[]
        for i in range(10):
            self.ten_prev_char.append(" ")


        for i in ascii_uppercase:
            self.ct[i] = 0

        print("Loaded model from disk")
        self.controller = controller
        # self.root = tk.Tk()
        # self.root.title("Sign Language To Text Conversion")
        # self.root.geometry("1500x800")

        self.panel = tk.Label(self)
        self.panel.place(x=100, y=3, width=480, height=640)

        self.panel2 = tk.Label(self)  # initialize image panel
        self.panel2.place(x=700, y=115, width=400, height=400)

        self.Title = tk.Label(self)
        self.Title.place(x=170, y=5)
        self.Title.config(text="Chuyển ngôn ngữ ký hiệu nói sang văn bản/giọng", font=("Courier", 30, "bold"))

        self.panel3 = tk.Label(self)  # Ký tự
        self.panel3.place(x=200, y=585)

        self.Character = tk.Label(self)
        self.Character.place(x=20, y=580)
        self.Character.config(text="Ký tự :", font=("Courier", 30, "bold"))

        self.panel5 = tk.Label(self)  # Câu văn
        self.panel5.place(x=140, y=632)

        self.Sentence = tk.Label(self)
        self.Sentence.place(x=20, y=632)
        self.Sentence.config(text="Câu :", font=("Courier", 30, "bold"))

        self.Suggestions = tk.Label(self)
        # self.Suggestions.place(x=10, y=700)
        self.Suggestions.config(text="Suggestions :", fg="red", font=("Courier", 30, "bold"))


        self.b1=tk.Button(self)
        # self.b1.place(x=390,y=700)

        self.b2 = tk.Button(self)
        # self.b2.place(x=590, y=700)

        self.b3 = tk.Button(self)
        # self.b3.place(x=790, y=700)

        self.b4 = tk.Button(self)
        # self.b4.place(x=990, y=700)

        self.back = tk.Button(self)
        self.back.place(x=1305, y=630)
        self.back.config(text="Trở lại", font=("Courier", 20), command=self.go_to_start_page)

        self.speak = tk.Button(self)
        self.speak.place(x=1235, y=630)
        self.speak.config(text="Nói", font=("Courier", 20), wraplength=70, command=self.speak_fun)

        self.clear = tk.Button(self)
        self.clear.place(x=1105, y=630)
        self.clear.config(text="Xóa hết", font=("Courier", 20), wraplength=120,command=self.clear_fun)

        self.delete_one = tk.Button(self)
        self.delete_one.place(x=1035, y=630)
        self.delete_one.config(text="Xóa", font=("Courier", 20), wraplength=70, command=self.delete_fun)





        self.str = " "
        self.ccc=0
        self.word = " "
        self.current_symbol = "C"
        self.photo = "Empty"


        self.word1=" "
        self.word2 = " "
        self.word3 = " "
        self.word4 = " "


        # self.video_loop()



        # self.start_webcam()
        # print(self.tes)



    def video_loop(self):
        if not self.camera_available or not self.video_loop_running:
            return
        try:
            ok, frame = self.vs.read()
            cv2image = cv2.flip(frame, 1)
            hands, frame = hd.findHands(cv2image, draw=False, flipType=True)
            cv2image_copy=np.array(cv2image)
            cv2image = cv2.cvtColor(cv2image, cv2.COLOR_BGR2RGB)
            self.current_image = Image.fromarray(cv2image)
            imgtk = ImageTk.PhotoImage(image=self.current_image)
            self.panel.imgtk = imgtk
            self.panel.config(image=imgtk)

            if hands:
                # #print(" --------- lmlist=",hands[1])
                hand = hands[0]
                x, y, w, h = hand['bbox']
                image = cv2image_copy[y - offset:y + h + offset, x - offset:x + w + offset]

                white = cv2.imread("white.jpg")
                # img_final=img_final1=img_final2=0

                handz, image = hd2.findHands(image, draw=False, flipType=True)
                print(" ", self.ccc)
                self.ccc += 1
                if handz:
                    hand = handz[0]
                    self.pts = hand['lmList']
                    # x1,y1,w1,h1=hand['bbox']

                    os = ((img_size - w) // 2) - 15
                    os1 = ((img_size - h) // 2) - 15
                    for t in range(0, 4, 1):
                        cv2.line(white, (self.pts[t][0] + os, self.pts[t][1] + os1), (self.pts[t + 1][0] + os, self.pts[t + 1][1] + os1),
                                 (0, 255, 0), 3)
                    for t in range(5, 8, 1):
                        cv2.line(white, (self.pts[t][0] + os, self.pts[t][1] + os1), (self.pts[t + 1][0] + os, self.pts[t + 1][1] + os1),
                                 (0, 255, 0), 3)
                    for t in range(9, 12, 1):
                        cv2.line(white, (self.pts[t][0] + os, self.pts[t][1] + os1), (self.pts[t + 1][0] + os, self.pts[t + 1][1] + os1),
                                 (0, 255, 0), 3)
                    for t in range(13, 16, 1):
                        cv2.line(white, (self.pts[t][0] + os, self.pts[t][1] + os1), (self.pts[t + 1][0] + os, self.pts[t + 1][1] + os1),
                                 (0, 255, 0), 3)
                    for t in range(17, 20, 1):
                        cv2.line(white, (self.pts[t][0] + os, self.pts[t][1] + os1), (self.pts[t + 1][0] + os, self.pts[t + 1][1] + os1),
                                 (0, 255, 0), 3)
                    cv2.line(white, (self.pts[5][0] + os, self.pts[5][1] + os1), (self.pts[9][0] + os, self.pts[9][1] + os1), (0, 255, 0),
                             3)
                    cv2.line(white, (self.pts[9][0] + os, self.pts[9][1] + os1), (self.pts[13][0] + os, self.pts[13][1] + os1), (0, 255, 0),
                             3)
                    cv2.line(white, (self.pts[13][0] + os, self.pts[13][1] + os1), (self.pts[17][0] + os, self.pts[17][1] + os1),
                             (0, 255, 0), 3)
                    cv2.line(white, (self.pts[0][0] + os, self.pts[0][1] + os1), (self.pts[5][0] + os, self.pts[5][1] + os1), (0, 255, 0),
                             3)
                    cv2.line(white, (self.pts[0][0] + os, self.pts[0][1] + os1), (self.pts[17][0] + os, self.pts[17][1] + os1), (0, 255, 0),
                             3)

                    for i in range(21):
                        cv2.circle(white, (self.pts[i][0] + os, self.pts[i][1] + os1), 2, (0, 0, 255), 1)

                    res=white
                    self.predict(res)

                    self.current_image2 = Image.fromarray(res)

                    imgtk = ImageTk.PhotoImage(image=self.current_image2)

                    self.panel2.imgtk = imgtk
                    self.panel2.config(image=imgtk)

                    self.panel3.config(text=self.current_symbol, font=("Courier", 30))

                    #self.panel4.config(text=self.word, font=("Courier", 30))



                    # self.b1.config(text=self.word1, font=("Courier", 20), wraplength=825, command=self.action1)
                    # self.b2.config(text=self.word2, font=("Courier", 20), wraplength=825,  command=self.action2)
                    # self.b3.config(text=self.word3, font=("Courier", 20), wraplength=825,  command=self.action3)
                    # self.b4.config(text=self.word4, font=("Courier", 20), wraplength=825,  command=self.action4)

            self.panel5.config(text=self.str, font=("Courier", 30), wraplength=1025)
        except Exception:
            print("Khong phat hien camera")
        finally:
            self.after(1, self.video_loop)


    def action1(self):
        idx_space = self.str.rfind(" ")
        idx_word = self.str.find(self.word, idx_space)
        last_idx = len(self.str)
        self.str = self.str[:idx_word]
        self.str = self.str + self.word1.upper()


    def action2(self):
        idx_space = self.str.rfind(" ")
        idx_word = self.str.find(self.word, idx_space)
        last_idx = len(self.str)
        self.str=self.str[:idx_word]
        self.str=self.str+self.word2.upper()
        #self.str[idx_word:last_idx] = self.word2


    def action3(self):
        idx_space = self.str.rfind(" ")
        idx_word = self.str.find(self.word, idx_space)
        last_idx = len(self.str)
        self.str = self.str[:idx_word]
        self.str = self.str + self.word3.upper()



    def action4(self):
        idx_space = self.str.rfind(" ")
        idx_word = self.str.find(self.word, idx_space)
        last_idx = len(self.str)
        self.str = self.str[:idx_word]
        self.str = self.str + self.word4.upper()


    def speak_fun(self):
        self.normalized_text = unicodedata.normalize("NFC", self.str)
        self.speak_engine.say(self.normalized_text)
        self.speak_engine.runAndWait()


    def clear_fun(self):
        self.str=" "
        self.word1 = " "
        self.word2 = " "
        self.word3 = " "
        self.word4 = " "

    def delete_fun(self):
        self.str= self.str[0:-1]
    def predict(self, test_image):
        white=test_image
        # white = white.reshape(1, img_size, img_size, 3)
        prediction, index = self.model.getPrediction(white, draw=False)
        # prob = np.array(self.model.predict(white)[0], dtype='float32')
        # ch1 = np.argmax(prob, axis=0)
        # prob[ch1] = 0

        chars_dau = ["A", "O", "U", "I", "Y", "\u0302", "\u031B"]
        chars_mu = ["A", "O", "E"]
        chars_rau = ["O", "U"]
        char_tontai = ["Backspace", "dấu mũ", "dấu râu"]

        if labels[index]=="tiếp tục" and self.prev_char!="tiếp tục":
            if self.ten_prev_char[(self.count-2)%10]!="tiếp tục":
                if self.ten_prev_char[(self.count-2)%10]=="Backspace":
                    self.str=self.str[0:-1]
                elif self.ten_prev_char[(self.count-2)%10]=="dấu cách":
                    self.str = self.str + " "
                elif self.ten_prev_char[(self.count-2)%10]=="S" and self.str[-1] in chars_dau:
                    self.str = self.str + "\u0301"
                elif self.ten_prev_char[(self.count-2)%10]=="B" and self.str[-1] in chars_dau:
                    self.str = self.str + "\u0300"
                elif self.ten_prev_char[(self.count-2)%10]=="R" and self.str[-1] in chars_dau:
                    self.str = self.str + "\u0309"
                elif self.ten_prev_char[(self.count-2)%10]=="C" and self.str[-1] in chars_dau:
                    self.str = self.str + "\u0323"
                elif self.ten_prev_char[(self.count-2)%10]=="X" and self.str[-1] in chars_dau:
                    self.str = self.str + "\u0303"
                elif self.ten_prev_char[(self.count-2)%10]=="dấu mũ" and self.str[-1] in chars_mu:
                    self.str = self.str + "\u0302"
                elif self.ten_prev_char[(self.count-2)%10]=="dấu râu" and self.str[-1] in chars_rau:
                    self.str = self.str + "\u031B"
                else:
                    if self.ten_prev_char[(self.count - 2) % 10] not in char_tontai:
                        self.str = self.str + self.ten_prev_char[(self.count-2)%10]
            else:
                if self.ten_prev_char[(self.count - 0) % 10] not in char_tontai:
                    self.str = self.str + self.ten_prev_char[(self.count - 0) % 10]






        self.prev_char=labels[index]
        self.current_symbol=labels[index]
        self.count += 1
        self.ten_prev_char[self.count%10]=labels[index]


        if len(self.str.strip())!=0:
            st=self.str.rfind(" ")
            ed=len(self.str)
            word=self.str[st+1:ed]
            self.word=word
            print("----------word = ",word)
            if len(word.strip())!=0:
                ddd.check(word)
                lenn = len(ddd.suggest(word))
                if lenn >= 4:
                    self.word4 = ddd.suggest(word)[3]

                if lenn >= 3:
                    self.word3 = ddd.suggest(word)[2]

                if lenn >= 2:
                    self.word2 = ddd.suggest(word)[1]

                if lenn >= 1:
                    self.word1 = ddd.suggest(word)[0]
            else:
                self.word1 = " "
                self.word2 = " "
                self.word3 = " "
                self.word4 = " "

    def go_to_start_page(self):

        self.off_webcam()
        self.controller.show_frame(StartPage)
        self.video_loop_running = False

    def start_webcam(self):
        self.camera_available = True
        self.video_loop_running = False

        self.vs = cv2.VideoCapture(0)
        if not self.vs.isOpened():
            print("Không thể mở camera.")
            self.camera_available = False
        else:
            self.video_loop_running = True
            self.video_loop()
        self.clear_fun()

    def off_webcam(self):
        self.vs.release()
        cv2.destroyAllWindows()
    def destructor(self):

        print("Closing Application...")
        print(self.ten_prev_char)
        self.destroy()
        self.vs.release()
        cv2.destroyAllWindows()


print("Starting Application...")

# (Application()).root.mainloop()
app = Tk_Manage()
app.geometry("1500x800")
app.mainloop()