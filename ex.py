from tkinter import * #used for gui
from tkinter import filedialog #used for selecting file from pc
import os   #executing the file
import cv2      #open_cv library
import numpy as np #very useful in working with large multidimensional arrays

root = Tk()
root.title('Cartoonify')
root.geometry('600x400')
root.config(bg='grey')
font_tuple = ("Times",18,"bold","underline","italic")

l = Label(root,text='CARTOONIFY IMAGE\nUSING MACHINE LEARNING')
l.config(bg='grey',fg='white',font=font_tuple)
l.pack(pady=20)



#read the image in the form of matrix
def read_img(filename):
    img = cv2.imread(filename)
    return img

def edge_detection(img,line_wdt,blur):#for edge detection
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)#grayscale
    grayBlur = cv2.medianBlur(gray,blur)#blur the grayscale to help in edge detection
    edges = cv2.adaptiveThreshold(grayBlur, 255, cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,line_wdt,blur)
    return edges

def color_quantisation(img,k):# color quantisation
    data = np.float32(img).reshape((-1,3))#matrix conversion
    criteria = (cv2.TermCriteria_EPS + cv2.TermCriteria_MAX_ITER, 20,0.001)
    ret,label, center  = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)#typedef into int
    result = center[label.flatten()]#change array into one dimension
    result = result.reshape(img.shape)#reshape back
    return result


def file_raw():
    global file_raw
    file_raw=filedialog.askopenfilename()
    # my_lable.config(text=file_raw)
    img = read_img(file_raw)
    line_wdt = 9
    blur_value = 7
    totalColors = 8

    edgeImg = edge_detection(img,line_wdt,blur_value)
    img = color_quantisation(img,totalColors)
    blurred = cv2.bilateralFilter(img,d=9,sigmaColor=150,sigmaSpace=150)
    cartoon = cv2.bitwise_and(blurred,blurred,mask=edgeImg)

    cv2.imwrite('cartoon.jpg',cartoon)


    os.system('cartoon.jpg')

# my_lable = Label(root,text="")
# my_lable.pack(pady=20)
choose_btn = PhotoImage(file='button.png')

choose = Button(root,image=choose_btn,command=file_raw,border=0,bg='grey')
choose.config(font="times 16 bold")
choose.pack(pady=20)

root.mainloop()


























# First we input the image
# then draw the sketch of the image
# then make the painting image  of the original image
# then merge both the images

#read_img
# get the grayscale image which is a black and white image of org image

#adaptive_threshold used in edge detection
#adaptive_threshold_mean->an adaptive thresholding method
#Threshold_binary->for every pixel same threshold is applied

#kmeans->unregistered learning algorithm
#divides unlabled dataset into k different clusters in such a way that each 
# dataset belongs only one group that has similar properties