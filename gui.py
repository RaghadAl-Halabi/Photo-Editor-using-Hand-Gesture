from tkinter import (ttk,Tk,PhotoImage,Canvas, filedialog, colorchooser,RIDGE,
                     GROOVE,ROUND,Scale,HORIZONTAL, Label, Frame)
import cv2 as cv
from PIL import ImageTk, Image
import numpy as np
import tkinter.messagebox
import customtkinter
import math
import time
import copy
from pynput.mouse import Button, Controller
import os

customtkinter.set_appearance_mode("Dark")  # Modes: "System" (standard), "Dark", "Light"
customtkinter.set_default_color_theme("blue") # Themes: "blue" (standard), "green", "dark-blue"
camera = cv.VideoCapture(0)

camera.set(10, 200)

fgbg = cv.createBackgroundSubtractorMOG2(0,50)

class FrontEnd(customtkinter.CTk):
    def splash_screen(self):
        w = Tk()

        width_of_window = 427
        height_of_window = 250
        screen_width = w.winfo_screenwidth()
        screen_height = w.winfo_screenheight()
        x_coordinate = (screen_width / 2) - (width_of_window / 2)
        y_coordinate = (screen_height / 2) - (height_of_window / 2)
        w.geometry("%dx%d+%d+%d" % (width_of_window, height_of_window, x_coordinate, y_coordinate))
        w.overrideredirect(1)  # for hiding titlebar
        Frame(w, width=427, height=250, bg='#272727').place(x=0, y=0)
        label1 = Label(w, text='PHOTO EDITOR', fg='white', bg='#272727')
        label1.configure(
            font=("Game Of Squids", 24, "bold"))
        label1.place(x=80, y=90)

        label2 = Label(w, text='Loading...', fg='white', bg='#272727')
        label2.configure(font=("Calibri", 11))
        label2.place(x=10, y=215)

        # making animation
        image_a = ImageTk.PhotoImage(Image.open('D://nlp projects/cvproject/img/c2.png'))
        image_b = ImageTk.PhotoImage(Image.open('D://nlp projects/cvproject/img/c1.png'))

        for i in range(5):
            l1 = Label(w, image=image_a, border=0, relief='sunken').place(x=180, y=145)
            l2 = Label(w, image=image_b, border=0, relief='sunken').place(x=200, y=145)
            l3 = Label(w, image=image_b, border=0, relief='sunken').place(x=220, y=145)
            l4 = Label(w, image=image_b, border=0, relief='sunken').place(x=240, y=145)
            w.update_idletasks()
            time.sleep(0.1)

            l1 = Label(w, image=image_b, border=0, relief='sunken').place(x=180, y=145)
            l2 = Label(w, image=image_a, border=0, relief='sunken').place(x=200, y=145)
            l3 = Label(w, image=image_b, border=0, relief='sunken').place(x=220, y=145)
            l4 = Label(w, image=image_b, border=0, relief='sunken').place(x=240, y=145)
            w.update_idletasks()
            time.sleep(0.1)

            l1 = Label(w, image=image_b, border=0, relief='sunken').place(x=180, y=145)
            l2 = Label(w, image=image_b, border=0, relief='sunken').place(x=200, y=145)
            l3 = Label(w, image=image_a, border=0, relief='sunken').place(x=220, y=145)
            l4 = Label(w, image=image_b, border=0, relief='sunken').place(x=240, y=145)
            w.update_idletasks()
            time.sleep(0.1)

            l1 = Label(w, image=image_b, border=0, relief='sunken').place(x=180, y=145)
            l2 = Label(w, image=image_b, border=0, relief='sunken').place(x=200, y=145)
            l3 = Label(w, image=image_b, border=0, relief='sunken').place(x=220, y=145)
            l4 = Label(w, image=image_a, border=0, relief='sunken').place(x=240, y=145)
            w.update_idletasks()
            time.sleep(0.1)

        self.upload_action()
        w.destroy()

    def __init__(self):
        self.splash_screen()
        super().__init__()
        # configure window
        self.title("Photo Editor")
        self.iconbitmap('D://nlp projects/cvproject/img/logo.ico')
        width = self.winfo_screenwidth()
        height = self.winfo_screenheight()
        self.geometry("%dx%d" % (width, height))
        self.state('zoomed')
        self.resizable(0, 0)
        self.bind('<Escape>', lambda e: mainWindow.quit())


        # configure grid layout
        self.grid_columnconfigure((3), weight=3)
        self.grid_columnconfigure((0, 1, 2), weight=1)
        self.grid_rowconfigure((1, 2), weight=1)


        #menu
        self.menu_frame = customtkinter.CTkFrame(self, width= width, height= 60)
        self.menu_frame.grid(row=0, column=0, columnspan=4, sticky="nsew")

        my_image1 = customtkinter.CTkImage(light_image=Image.open("D://nlp projects/cvproject/img/1.png"),
                                          dark_image=Image.open("D://nlp projects/cvproject/img/1.png"),
                                          size=(40, 40))
        self.menu_button_1 = customtkinter.CTkButton(self.menu_frame, fg_color=("#dbdbdb","#2b2b2b"),
                                                        text_color= ("#2b2b2b", "white"),
                                                        image=my_image1,
                                                        hover_color="#C689C6", text= 'Editing',command=self.Editing_action) #command=self.sidebar_button_event
        self.menu_button_1.grid(row=0, column=0, padx=20, pady=10, sticky="nsew")
        my_image2 = customtkinter.CTkImage(light_image=Image.open("D://nlp projects/cvproject/img/3.png"),
                                           dark_image=Image.open("D://nlp projects/cvproject/img/3.png"),
                                           size=(40, 40))
        self.menu_button_2 = customtkinter.CTkButton(self.menu_frame, fg_color=("#dbdbdb","#2b2b2b"),
                                                     image=my_image2,
                                                     text_color= ("#2b2b2b", "white"),
                                                        hover_color="#7FB77E", text= 'Perspective transform', command=self.Perspective_action)
        self.menu_button_2.grid(row=0, column=1, padx=20, pady=10)
        my_image3 = customtkinter.CTkImage(light_image=Image.open("D://nlp projects/cvproject/img/4.png"),
                                           dark_image=Image.open("D://nlp projects/cvproject/img/4.png"),
                                           size=(40, 40))
        self.menu_button_3 = customtkinter.CTkButton(self.menu_frame, fg_color=("#dbdbdb","#2b2b2b"),
                                                     image=my_image3,
                                                     text_color= ("#2b2b2b", "white"),
                                                        hover_color="#6E85B7", text= 'Painting'
                                                     , command=self.Painting_action)
        self.menu_button_3.grid(row=0, column=2, padx=20, pady=10)

        #sidebar
        self.sidebar_frame = customtkinter.CTkFrame(self, corner_radius=5)
        self.sidebar_frame.grid(row=1, column=0, rowspan=2, sticky="nsew", pady=10 ,padx=5)
        self.sidebar_frame.grid_propagate(False)

        # canvas
        self.canvas_frame = customtkinter.CTkFrame(self, corner_radius=0)
        self.canvas_frame.grid(column=1 , row=1, rowspan=2, columnspan=2, sticky="NSEW", pady=10, padx=5)
        self.canvas_frame.grid_propagate(False)
        self.canvas_frame.grid_columnconfigure((0,2), weight=1)
        self.canvas_frame.grid_rowconfigure((0,2), weight=1)
        self.canvas= Canvas(self.canvas_frame, bg='#dbdbdb', width=626, height=968)
        self.canvas.grid( column=1 , row=1, sticky='NSEW')
        self.canvas.grid_anchor('center')
        self.button = customtkinter.CTkButton(self.canvas_frame, fg_color=("#dbdbdb", "#2b2b2b"),
                                                     text_color=("#2b2b2b", "white"),
                                                     hover_color="#6E85B7", text='Apply'
                                                     , command=self.apply_action)
        self.button.grid( column=1 , row=2, sticky='s')
        self.button.grid_anchor('center')

        my_imagesb2 = customtkinter.CTkImage(light_image=Image.open("D://nlp projects/cvproject/img/16.png"),
                                            dark_image=Image.open("D://nlp projects/cvproject/img/16.png"),
                                            size=(40, 40))
        self.button2 = customtkinter.CTkButton(self.canvas_frame, fg_color=("#dbdbdb", "#2b2b2b"),
                                               image=my_imagesb2,
                                               compound="right",
                                              text_color=("#2b2b2b", "white"),
                                              hover_color="#6E85B7", text='Save'
                                              , command=self.save_action)
        self.button2.grid(column=2, row=2, sticky='s')
        self.button2.grid_anchor('center')


        # sidebar_right
        self.sidebarright_frame = customtkinter.CTkFrame(self)
        self.sidebarright_frame.grid(row=1, column=3, rowspan=2, columnspan=2, sticky="NSEW", pady=10, padx=5)
        self.sidebarright_label = customtkinter.CTkLabel(self.sidebarright_frame,width=780,  text="")
        self.sidebarright_label.place(relx=0.5, rely=0.5, anchor=tkinter.CENTER)


        self.display_image(image= self.edited_image)
        self.open_camera()

    '''
    المعالجة للفديو هون
    '''
    def open_camera(self):
        ret, frame = camera.read()
        frame = cv.flip(frame, 1)
        cv.imshow("original", frame)

        ''' 
        عرض الفريم على الواجهة
        '''
        # Convert image from one color space to other
        opencv_image = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        # Capture the latest frame and transform to image
        captured_image = Image.fromarray(opencv_image)
        # Convert captured image to photoimage
        photo_image = ImageTk.PhotoImage(image=captured_image)
        photo_image = customtkinter.CTkImage(light_image=captured_image, dark_image=captured_image,
                                             size=(frame.shape[1], frame.shape[0]))
        # Configure image in the label
        self.sidebarright_label.configure(image=photo_image)
        self.update()
        '''
        خلص
        '''


        # Converting from gbr to YCbCr color space
        img_YCrCb = cv.cvtColor(frame, cv.COLOR_BGR2YCrCb)
        # cv.imshow("img_YCrCb", img_YCrCb)

        '''''
        ForegroundBackground model
        '''''
        # Splitting the channels in order to deal only with the y channel that represents the illumination
        y, cr, cb = cv.split(img_YCrCb)
        # cv.imshow("y", y)

        # Bilateral filtering in order to smooth the frame with the preserving of the edges
        blur = cv.bilateralFilter(y, 5, 50, 100)

        # Adaptive thresholding in order to give better results for videos with varying illumination
        thresholdedImg = cv.adaptiveThreshold(blur, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                                              cv.THRESH_BINARY_INV, 15, 3)
        cv.imshow('thresholdedImg', thresholdedImg)

        # Background Removal based on the moving objects
        fgbg.setDetectShadows(False)
        fgmask = fgbg.apply(thresholdedImg, 0)
        foreground = cv.bitwise_and(thresholdedImg, thresholdedImg, mask=fgmask)
        cv.imshow('foreground', foreground)

        # Removing the noise (the white dots that are considered as moving elements in the frames) with MORPH_OPEN
        opening = cv.morphologyEx(foreground, cv.MORPH_OPEN, np.ones((3, 3), np.uint8))
        # cv.imshow("opening", opening)
        # contours2, _ = cv.findContours(opening, cv.RETR_TREE,
        #                                cv.CHAIN_APPROX_SIMPLE)
        # openingc = cv.drawContours(opening, contours1, -1, (0, 255, 0), 1)
        # cv.imshow('openingc', openingc)

        # Adding more weight for the foreground with DILATION
        kernel = np.ones((9, 9), np.uint8)
        dilation = cv.dilate(opening, kernel, iterations=1)
        # cv.imshow("dilation", dilation)
        # contours2, _ = cv.findContours(opening, cv.RETR_TREE,
        #                                cv.CHAIN_APPROX_SIMPLE)
        # dilationc = cv.drawContours(dilation, contours1, -1, (0, 255, 0), 1)
        # cv.imshow('dilationc', dilationc)

        '''''
            Skin model
        '''''
        # Equalizing histogram of the y channel to effectively spread out the most frequent intensity values of illumination
        Image_equ = cv.equalizeHist(y)
        # cv.imshow("Image_equalized", Image_equ)

        # Getting back the image in the YCrCb space
        image_merge = cv.merge([y, cr, cb])
        cv.imshow('image_merge', image_merge)

        # The values ranges that represent the skin colors in the YCrCb space
        skin_mask_mint = np.array((0, 133, 77))
        skin_mask_maxt = np.array((255, 173, 127))

        # Creating the skin mask using the previous ranges
        skin_mask = cv.inRange(image_merge, skin_mask_mint, skin_mask_maxt)
        thresholdedSkin = cv.adaptiveThreshold(skin_mask, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                                               cv.THRESH_BINARY_INV, 15, 4)
        cv.imshow('thresholdedSkin', thresholdedImg)

        # Applying MORPH_OPEN then MORPH_CLOSE to enhance the skin mask
        skin_mask = cv.morphologyEx(skin_mask, cv.MORPH_OPEN, np.ones((3, 3), np.uint8))
        skin_mask = cv.morphologyEx(skin_mask, cv.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        cv.imshow('skin_mask', skin_mask)

        # # FOR LATER USE
        # edges = cv.Canny(skin_mask, 100, 200)
        # # cv.imshow('edges', edges)

        # Non-skin Removal based on the skin mask on the ForegroundBackground model
        skinForeground = cv.bitwise_and(dilation, skin_mask)
        cv.imshow('skinForeground', skinForeground)

        # contours1, _ = cv.findContours(skinForeground, cv.RETR_TREE,
        #                                cv.CHAIN_APPROX_SIMPLE)
        # skinForegroundc = cv.drawContours(skinForeground, contours1, -1, (0, 255, 0), 1)
        # cv.imshow('skinForegroundc', skinForegroundc)
        #
        image = cv.resize(skinForeground, dsize=(720, 720),
                          interpolation=cv.INTER_NEAREST_EXACT)
        cv.imshow('image', image)

        # Getting the contours and convex hull
        skinMask1 = copy.deepcopy(image)
        contours, hierarchy = cv.findContours(foreground, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        length = len(contours)
        maxArea = -1
        if length > 0:
            for i in range(length):
                temp = contours[i]
                area = cv.contourArea(temp)
                if area > maxArea:
                    maxArea = area
                    ci = i
                    res = contours[ci]
                    hull = cv.convexHull(res)
                    drawing = np.zeros(frame.shape, np.uint8)
                    cv.drawContours(drawing, [res], 0, (0, 255, 0), 2)
                    cv.drawContours(drawing, [hull], 0, (0, 0, 255), 3)

                    # isFinishCal, cnt = calculateFingers(res, drawing)
                    # print
                    # "Fingers", cnt
                    cv.imshow('output', drawing)
        k = cv.waitKey(1)
        if k % 256 == 27:
            # ESC pressed
            print("Escape hit, closing...")

            camera.release()
            cv.destroyAllWindows()

        '''
        هاد السطر تحديدا بدل الـ while ليكرر التابع
        '''
        # Repeat the same process after every x seconds
        self.sidebarright_label.after(10, self.open_camera)

    def upload_action(self):

        self.filename = filedialog.askopenfilename()
        self.original_image = cv.imread(self.filename)

        self.edited_image = cv.imread(self.filename)
        self.filtered_image = cv.imread(self.filename)

    def refresh_side_frame(self):
        try:
            for widget in self.sidebar_frame.winfo_children():
                widget.destroy()

        except:
            pass
        self.sidebar_frame.grid_rowconfigure((1), weight=0)
        self.sidebar_frame.grid_rowconfigure((5), weight=0)
        self.sidebar_frame.grid_rowconfigure((9), weight=0)
        self.canvas.unbind("<ButtonPress>")
        self.canvas.unbind("<B1-Motion>")
        self.canvas.unbind("<ButtonRelease>")
        # self.display_image(self.edited_image)
        # self.sidebar_frame = customtkinter.CTkFrame(self, corner_radius=5)
        # self.sidebar_frame.grid(row=1, column=0, rowspan=2, sticky="nsew")

    def Editing_action(self, event =None):
        self.refresh_side_frame()
        self.sidebar_frame.grid_columnconfigure(0, weight=1)

        self.label = customtkinter.CTkLabel(self.sidebar_frame, text="Editing",
                                                 fg_color="#C689C6",
                                                 height=40,
                                                 corner_radius=5,
                                                 anchor="center",
                                                 font=customtkinter.CTkFont(size=20, weight="bold"))
        self.label.grid(row=0, column=0,sticky="nsew")

        #Rotate
        my_imager = customtkinter.CTkImage(light_image=Image.open("D://nlp projects/cvproject/img/9.png"),
                                           dark_image=Image.open("D://nlp projects/cvproject/img/9.png"),
                                           size=(50, 50))
        self.rotate = customtkinter.CTkLabel(self.sidebar_frame, text="Rotate",
                                           image=my_imager,
                                           compound= "right",
                                           corner_radius=5,
                                           anchor="center",
                                           font=customtkinter.CTkFont(size=18, weight="bold"))
        self.rotate.grid(row=4, column=0, pady=20, sticky="nsew")
        self.slider_rotate = customtkinter.CTkSlider(self.sidebar_frame, from_=0, to=360, number_of_steps=36, command=self.rotate_action,
                                                     button_color="#C689C6",
                                                     button_hover_color="#C689C6",
                                                     )
        self.slider_rotate.set(output_value=0)
        self.slider_rotate.grid(row=5, column=0, padx=(20, 10), pady=(10, 10), sticky="ew")

        #Scale
        my_images = customtkinter.CTkImage(light_image=Image.open("D://nlp projects/cvproject/img/6.png"),
                                           dark_image=Image.open("D://nlp projects/cvproject/img/6.png"),
                                           size=(50, 50))
        self.scale = customtkinter.CTkLabel(self.sidebar_frame, text="Scale",
                                           image=my_images,
                                           compound="right",
                                           corner_radius=5,
                                           anchor="center",
                                           font=customtkinter.CTkFont(size=18, weight="bold"))
        self.scale.grid(row=6, column=0, pady=20, sticky="nsew")
        self.slider_scale = customtkinter.CTkSlider(self.sidebar_frame, from_=10, to=200, number_of_steps=20,
                                                    button_color="#C689C6",
                                                    button_hover_color="#C689C6",
                                                     command=self.scale_action)
        self.slider_scale.set(output_value=100)
        self.slider_scale.grid(row=7, column=0, padx=(20, 10), pady=(10, 10), sticky="ew")

        #translation
        my_imaget = customtkinter.CTkImage(light_image=Image.open("D://nlp projects/cvproject/img/8.png"),
                                           dark_image=Image.open("D://nlp projects/cvproject/img/8.png"),
                                           size=(50, 50))
        self.trans = customtkinter.CTkLabel(self.sidebar_frame, text="Translate",
                                           image=my_imaget,
                                           compound="right",
                                           corner_radius=5,
                                           anchor="center",
                                           font=customtkinter.CTkFont(size=18, weight="bold"))
        self.trans.grid(row=8, column=0, pady=(20,0), sticky="nsew")

        self.switch_var = customtkinter.BooleanVar(value=False)
        switch_1 = customtkinter.CTkSwitch(self.sidebar_frame, text="", command=self.translation_action,

                                           width=100,
                                           font=customtkinter.CTkFont(size=18, weight="bold"),
                                           button_hover_color= "#C689C6",
                                           progress_color= "#C689C6",
                                           variable=self.switch_var, onvalue=True, offvalue=False)

        switch_1.grid(row=9, column=0, pady=30, sticky="ns")

    def translation_action(self):
        if self.switch_var.get():
            self.canvas.bind("<ButtonPress>", self.start_translation)
            self.canvas.bind("<B1-Motion>", self.translation)

    def start_translation(self, event=None):
        if self.switch_var.get():
            self.translation_start_x = event.x
            self.translation_start_y = event.y

    def translation(self, event=None):
        if self.switch_var.get():
            image = self.edited_image.copy()
            num_rows, num_cols = self.edited_image.shape[:2]
            self.translation_matrix = np.float32([[1, 0, event.x* self.ratio-(num_cols/2) ], [0, 1, event.y* self.ratio-(num_rows/2)]])
            self.img_translation = cv.warpAffine(self.edited_image, self.translation_matrix, (num_cols , num_rows))
            self.display_image(image= self.filtered_image)
            self.filtered_image = self.img_translation
            # cv.imshow('ed', self.edited_image)
            # cv.imshow('fi', self.filtered_image)
            self.translation_start_x = event.x
            self.translation_start_y = event.y

    def rotate_action(self, value):
        num_rows, num_cols = self.edited_image.shape[:2]
        image_center = (num_cols / 2,  num_rows/ 2)
        self.rotation_matrix = cv.getRotationMatrix2D((image_center[0], image_center[1]), value, 1.0)
        abs_cos = np.abs(self.rotation_matrix[0, 0])
        abs_sin = np.abs(self.rotation_matrix[0, 1])
        bound_w = int((num_rows * abs_sin) + (num_cols * abs_cos))
        bound_h = int((num_rows * abs_cos) + (num_cols * abs_sin))
        self.rotation_matrix [0, 2] += (bound_w / 2 )- image_center[0]
        self.rotation_matrix [1, 2] += (bound_h / 2) - image_center[1]
        self.img_rotation  = cv.warpAffine(self.edited_image, self.rotation_matrix , (bound_w, bound_h))
        self.filtered_image= self.img_rotation
        self.display_image(image=self.filtered_image)

    def scale_action(self, value):
        self.img_scaled = cv.resize(self.edited_image, None, fx=value/100, fy=value/100, interpolation=cv.INTER_LINEAR)
        self.filtered_image = self.img_scaled
        self.display_image(image=self.filtered_image)

    def Perspective_action(self):
        self.refresh_side_frame()
        self.sidebar_frame.grid_columnconfigure((0), weight=1)
        self.label = customtkinter.CTkLabel(self.sidebar_frame, text="Perspective transform",
                                                 fg_color="#7FB77E",
                                                 height=40,
                                                 corner_radius=5,
                                                 anchor="center",
                                                 font=customtkinter.CTkFont(size=20, weight="bold"))
        self.label.grid(row=0, column=0,sticky="nsew")
        #warp
        my_imagew = customtkinter.CTkImage(light_image=Image.open("D://nlp projects/cvproject/img/10.png"),
                                           dark_image=Image.open("D://nlp projects/cvproject/img/10.png"),
                                           size=(50, 50))

        self.warp = customtkinter.CTkLabel(self.sidebar_frame, text="wrap",
                                           image=my_imagew,
                                           compound="right",
                                            corner_radius=5,
                                            anchor="center",
                                            font=customtkinter.CTkFont(size=18, weight="bold"))
        self.warp.grid(row=4, column=0, pady=20,sticky="nsew")
        self.radio_var = tkinter.IntVar(value=0)
        self.radio_button_1 = customtkinter.CTkRadioButton(master=self.sidebar_frame, variable=self.radio_var,
                                                           text="Vertical wave",
                                                           hover_color="#7FB77E",
                                                           fg_color="#7FB77E",
                                                           value=0, command=self.wrap_action)
        self.radio_button_1.grid(row=6, column=0, pady=10, padx=20, sticky="w")
        self.radio_button_2 = customtkinter.CTkRadioButton(master=self.sidebar_frame, variable=self.radio_var,
                                                           text="Horizontal wave",
                                                           hover_color="#7FB77E",
                                                           fg_color="#7FB77E",
                                                           value=1, command=self.wrap_action)
        self.radio_button_2.grid(row=7, column=0, pady=10, padx=20, sticky="w")
        self.radio_button_3 = customtkinter.CTkRadioButton(master=self.sidebar_frame, variable=self.radio_var,
                                                           text="Multidirectional wave",
                                                           hover_color="#7FB77E",
                                                           fg_color="#7FB77E",
                                                           value=2, command=self.wrap_action)
        self.radio_button_3.grid(row=9, column=0, pady=10, padx=20, sticky="w")
        self.radio_button_4 = customtkinter.CTkRadioButton(master=self.sidebar_frame, variable=self.radio_var,
                                                           text="Concave wave",
                                                           hover_color="#7FB77E",
                                                           fg_color="#7FB77E",
                                                           value=3, command=self.wrap_action)
        self.radio_button_4.grid(row=10, column=0, pady=10, padx=20, sticky="w")

        self.skew = customtkinter.CTkLabel(self.sidebar_frame, text="skew",
                                            corner_radius=5,
                                            anchor="center",
                                            font=customtkinter.CTkFont(size=18, weight="bold"))
        self.skew.grid(row=11, column=0, pady=20, sticky="nsew")

        my_imagesx = customtkinter.CTkImage(light_image=Image.open("D://nlp projects/cvproject/img/10.png"),
                                           dark_image=Image.open("D://nlp projects/cvproject/img/10.png"),
                                           size=(50, 50))
        self.skewx = customtkinter.CTkLabel(self.sidebar_frame, text="skewX",
                                            image=my_imagesx,
                                            compound="right",
                                           corner_radius=5,
                                            anchor="w",
                                           font=customtkinter.CTkFont(size=14, weight="bold"))
        self.skewx.grid(row=12, column=0, pady=10, sticky="nsew")
        self.slider_skewx = customtkinter.CTkSlider(self.sidebar_frame, from_=0, to=30, number_of_steps=30,
                                                    button_color="#7FB77E",
                                                    button_hover_color="#7FB77E",
                                                     command=self.skewx_action)
        self.slider_skewx.set(output_value=0)
        self.slider_skewx.grid(row=13, column=0, padx=(20, 10), pady=(10, 10), sticky="ew")

        my_imagesy = customtkinter.CTkImage(light_image=Image.open("D://nlp projects/cvproject/img/11.png"),
                                            dark_image=Image.open("D://nlp projects/cvproject/img/11.png"),
                                            size=(50, 50))
        self.skewy = customtkinter.CTkLabel(self.sidebar_frame, text="skewY",
                                            image=my_imagesy,
                                            compound="right",
                                            corner_radius=5,

                                            anchor="w",
                                            font=customtkinter.CTkFont(size=14, weight="bold"))
        self.skewy.grid(row=14, column=0, pady=10, sticky="nsew")
        self.slider_skewy = customtkinter.CTkSlider(self.sidebar_frame, from_=0, to=30, number_of_steps=30,
                                                    button_color="#7FB77E",
                                                    button_hover_color="#7FB77E",
                                                     command=self.skewy_action)
        self.slider_skewy.set(output_value=0)
        self.slider_skewy.grid(row=16, column=0, padx=(20, 10), pady=(10, 10), sticky="ew")

    def wrap_action(self):
        if self.radio_var.get() == 0:
            self.Vertical_wave()
        elif self.radio_var.get() == 1:
            self.Horizontal_wave()
        elif self.radio_var.get() == 2:
            self.Multidirectional_wave()
        else:
            self.Concave_wave()

    def skewx_action(self,value):
        num_rows, num_cols = self.edited_image.shape[:2]
        M = np.float32([[1, value/5, 0],
                        [0, 1, 0],
                        [0, 0, 1]])

        max_row_shift= value/5 * num_rows
        # if value<0:
        #     max_row_shift = -value/10 * num_rows
        self.sheared_img = cv.warpPerspective(self.edited_image, M,
                                              (int(max_row_shift+num_cols), int(num_rows)))
        self.filtered_image = self.sheared_img

        self.display_image(image=self.filtered_image)

    def skewy_action(self,value):
        num_rows, num_cols = self.edited_image.shape[:2]
        M = np.float32([[1, 0, 0],
                        [value/5, 1, 0],
                        [0, 0, 1]])
        max_col_shift = value / 5 * num_cols
        self.sheared_img = cv.warpPerspective( self.edited_image, M, (int(num_cols ), int(num_rows+max_col_shift )))
        self.filtered_image = self.sheared_img
        self.display_image(image=self.filtered_image)

    def Vertical_wave(self):
        rows, cols = self.edited_image.shape[:2]
        self.img_output = np.zeros(self.edited_image.shape, dtype=self.filtered_image.dtype)
        for i in range(rows):
            for j in range(cols):
                offset_x = int(25.0 * math.sin(2 * 3.14 * i / 180))
                offset_y = 0
                if j+offset_x < rows:
                    self.img_output[i,j] = self.edited_image[i,(j+offset_x)%cols]
                else:
                    self.img_output[i,j] = 0
        self.filtered_image= self.img_output
        self.display_image(image=self.filtered_image)

    def Horizontal_wave(self):
        rows, cols = self.edited_image.shape[:2]
        self.img_output = np.zeros(self.edited_image.shape, dtype=self.filtered_image.dtype)
        for i in range(rows):
            for j in range(cols):
                offset_x = 0
                offset_y = int(16.0 * math.sin(2 * 3.14 * j / 150))
                if i+offset_y < rows:
                    self.img_output[i,j] = self.edited_image[(i+offset_y)%rows,j]
                else:
                    self.img_output[i,j] = 0
        self.filtered_image = self.img_output
        self.display_image(image=self.filtered_image)

    def Multidirectional_wave(self):
        rows, cols = self.edited_image.shape[:2]
        self.img_output = np.zeros(self.edited_image.shape, dtype=self.filtered_image.dtype)
        for i in range(rows):
            for j in range(cols):
                offset_x = int(20.0 * math.sin(2 * 3.14 * i / 150))
                offset_y = int(20.0 * math.cos(2 * 3.14 * j / 150))
                if i + offset_y < rows and j + offset_x < cols:
                    self.img_output[i, j] = self.edited_image[(i + offset_y) % rows, (j + offset_x) % cols]
                else:
                    self.img_output[i, j] = 0
        self.filtered_image = self.img_output
        self.display_image(image=self.filtered_image)

    def Concave_wave(self):
        rows, cols = self.edited_image.shape[:2]
        self.img_output = np.zeros(self.edited_image.shape, dtype=self.filtered_image.dtype)
        for i in range(rows):
            for j in range(cols):
                offset_x = int(128.0 * math.sin(2 * 3.14 * i / (2 * cols)))
                offset_y = 0
                if j+offset_x < cols:
                    self.img_output[i, j] = self.edited_image[i,(j+offset_x)%cols]
                else:
                    self.img_output[i, j] = 0
        self.filtered_image = self.img_output
        self.display_image(image=self.filtered_image)

    def Painting_action(self, event=None):
        self.refresh_side_frame()
        self.sidebar_frame.grid_columnconfigure((0), weight=1)
        self.sidebar_frame.grid_rowconfigure((1), weight=1)
        self.sidebar_frame.grid_rowconfigure((5), weight=1)
        self.sidebar_frame.grid_rowconfigure((9), weight=7)
        self.label = customtkinter.CTkLabel(self.sidebar_frame, text="Painting",
                                            fg_color="#6E85B7",
                                            height=40,
                                            corner_radius=5,
                                            anchor="center",
                                            font=customtkinter.CTkFont(size=20, weight="bold"))
        self.label.grid(row=0, column=0, sticky="nsew")
        self.color_code = ((255, 0, 0), '#ff0000')

        height, width = self.filtered_image.shape[:2]
        self.imgCanvas = np.zeros((height, width, 3), np.uint8)
        self.draw_img = self.filtered_image.copy()

        self.canvas.bind("<ButtonPress>", self.start_draw)
        self.canvas.bind("<B1-Motion>", self.draw)

        my_imagep = customtkinter.CTkImage(light_image=Image.open("D://nlp projects/cvproject/img/14.png"),
                                            dark_image=Image.open("D://nlp projects/cvproject/img/14.png"),
                                            size=(40, 40))
        self.draw_color_button = customtkinter.CTkButton(
            self.sidebar_frame, text="Pick A Color",
            image=my_imagep,
            compound="right",
            hover_color="#6E85B7",
            fg_color=("#dbdbdb","#2b2b2b"), command=self.choose_color)
        self.draw_color_button.grid(
            row=2, column=0, padx=5, pady=5, sticky='s')
        self.brushThickness  = customtkinter.CTkLabel(self.sidebar_frame, text="brush Thickness ",
                                            corner_radius=5,
                                            anchor="w",
                                            font=customtkinter.CTkFont(size=14, weight="bold"))
        self.brushThickness .grid(row=3, column=0, pady=10, sticky="nsew")
        self.brushThickness_var = tkinter.IntVar(value=5)
        self.slider_brushThickness  = customtkinter.CTkSlider(self.sidebar_frame, from_=5, to=100, number_of_steps=10,
                                                    button_color="#6E85B7",
                                                    button_hover_color="#6E85B7",
                                                    variable=self.brushThickness_var)
        self.slider_brushThickness.set(output_value=0)
        self.slider_brushThickness.grid(row=4, column=0, padx=(20, 10), pady=(10, 10), sticky="ew")

        my_imageer = customtkinter.CTkImage(light_image=Image.open("D://nlp projects/cvproject/img/13.png"),
                                           dark_image=Image.open("D://nlp projects/cvproject/img/13.png"),
                                           size=(50, 50))
        self.eraser = customtkinter.CTkLabel(self.sidebar_frame, text="Eraser",
                                            image=my_imageer,
                                            compound="right",
                                            corner_radius=5,
                                            anchor="center",
                                            font=customtkinter.CTkFont(size=18, weight="bold"))
        self.eraser.grid(row=5, column=0, pady=(20, 0), sticky="nsew")
        self.eraser_var = customtkinter.BooleanVar(value=False)
        switch_eraser = customtkinter.CTkSwitch(self.sidebar_frame, text="",
                                           width=100,
                                           font=customtkinter.CTkFont(size=18, weight="bold"),
                                           button_hover_color="#6E85B7",
                                           progress_color="#6E85B7",
                                           variable=self.eraser_var, onvalue=True, offvalue=False)

        switch_eraser.grid(row=6, column=0, pady=(0, 30), sticky="ns")
        self.eraserThickness  = customtkinter.CTkLabel(self.sidebar_frame, text="eraser Thickness ",
                                                     corner_radius=5,
                                                     anchor="w",
                                                     font=customtkinter.CTkFont(size=14, weight="bold"))
        self.eraserThickness .grid(row=7, column=0, pady=10, sticky="nsew")
        self.eraserThickness_var = tkinter.IntVar(value=5)
        self.slider_eraserThickness = customtkinter.CTkSlider(self.sidebar_frame, from_=25, to=100, number_of_steps=7,
                                                             button_color="#6E85B7",
                                                             button_hover_color="#6E85B7",
                                                             variable=self.eraserThickness_var)
        self.slider_eraserThickness.set(output_value=5)
        self.slider_eraserThickness.grid(row=8, column=0, padx=(0, 10), pady=(10, 10), sticky="ew")

    def choose_color(self):
        self.color_code = colorchooser.askcolor(title="Choose color")

    def start_draw(self, event):
        self.x = event.x
        self.y = event.y
        self.draw_ids = []

    def draw(self, event):
        if self.eraser_var.get() :
            # cv.circle(self.filtered_image, (int(event.x * self.ratio), int(event.y * self.ratio)), self.eraserThickness_var.get(), self.color_code[0])
            # cv.circle(self.filtered_image,  (int(self.x * self.ratio), int(self.y * self.ratio)), 15, (255, 255, 255), cv.FILLED)
            cv.line(self.draw_img, (int(self.x * self.ratio), int(self.y * self.ratio)),
                    (int(event.x * self.ratio), int(event.y * self.ratio)),
                    (0, 0, 0), thickness=self.eraserThickness_var.get())
            cv.line(self.imgCanvas, (int(self.x * self.ratio), int(self.y * self.ratio)),
                    (int(event.x * self.ratio), int(event.y * self.ratio)),
                    (0, 0, 0), thickness=self.eraserThickness_var.get())
        else:
            # cv.circle(self.filtered_image, (int(event.x * self.ratio), int(event.y * self.ratio)), self.brushThickness_var.get(), self.color_code[0])
            self.draw_ids.append(self.canvas.create_line(self.x, self.y, event.x, event.y, width=5,
                                                         fill=self.color_code[-1], capstyle=ROUND, smooth=True))

            cv.line(self.draw_img, (int(self.x * self.ratio), int(self.y * self.ratio)),
                    (int(event.x * self.ratio), int(event.y * self.ratio)),
                    self.color_code[0], thickness=self.brushThickness_var.get())
            cv.line(self.imgCanvas, (int(self.x * self.ratio), int(self.y * self.ratio)),
                    (int(event.x * self.ratio), int(event.y * self.ratio)),
                    self.color_code[0], thickness=self.brushThickness_var.get())

        imgGray = cv.cvtColor(self.imgCanvas, cv.COLOR_BGR2GRAY)
        _, imgInv = cv.threshold(imgGray, 1, 255, cv.THRESH_BINARY_INV)
        imgInv = cv.cvtColor(imgInv, cv.COLOR_GRAY2RGB)
        self.filtered_image = cv.bitwise_or(self.draw_img, self.imgCanvas)
        self.display_image(image=self.filtered_image)
        self.draw_img = cv.bitwise_and(self.edited_image, imgInv)




        self.x = event.x
        self.y = event.y

    def apply_action(self):
        self.edited_image = self.filtered_image
        self.display_image(self.edited_image)
        self.slider_rotate.set(output_value=0)
        self.slider_scale.set(output_value=100)

    def cancel_action(self):
        self.display_image(self.edited_image)

    def save_action(self):
        original_file_type = self.filename.split('.')[-1]
        filename = filedialog.asksaveasfilename()
        filename = filename + "." + original_file_type

        save_as_image = self.edited_image
        cv.imwrite(filename, save_as_image)
        self.filename = filename

    def display_image(self, image=None):
        self.canvas.delete("all")
        if image is None:
            image = self.edited_image.copy()
        else:
            image = image

        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        height, width, channels = image.shape
        ratio = height / width

        new_width = width
        new_height = height
        if height > 630 or width > 620:
            if ratio < 1:
                new_width = 620
                new_height = int(new_width * ratio)
            else:
                new_height = 630
                new_width = int(new_height * (width / height))

        self.ratio = height / new_height

        self.new_image = cv.resize(image, (new_width, new_height))
        self.new_image = ImageTk.PhotoImage(
            Image.fromarray(self.new_image))
        self.canvas.config(width= new_width-4, height= new_height-4)
        self.canvas.create_image(
            new_width /2, new_height/2 , image =self.new_image, anchor="center")


mainWindow = FrontEnd()

mainWindow.mainloop()