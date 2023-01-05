import matplotlib.pyplot as plt
import tensorflow as tf
import tkinter as tk
import numpy as np
import cv2 as cv

root = tk.Tk()
img = np.zeros((100,100,1))
model = tf.keras.models.load_model('mnist_labelled.h5')
pred_label = tk.Label(root , text="Prediction: ")
canvas = tk.Canvas(root, height=100, width=100, bg='white')

def preprocessing(img):
    img = tf.image.resize(img , (100 , 100) , method=tf.image.ResizeMethod.NEAREST_NEIGHBOR, preserve_aspect_ratio=True).numpy()
    img = np.expand_dims(img , axis=2)
    img = np.expand_dims(img , axis=0)
    img = img / 255.0
    return img

def Draw(event):
    x,y = event.x , event.y

    canvas.create_rectangle(x-2,y-2,x+2,y+2,fill='#000000')
    img[y-2:y+2,x-2:x+2,0] = 255.0

def predict(img):
    
    img = preprocessing(img)
    val , label = model.predict(img)

    x , y , h , w = (label[0] * 100).astype('uint8').tolist()
    (x1 , y1) = (x - w//2,y + h//2)
    (x2 , y2) = (x + h//2,y - w//2)

    canvas.create_line(x1 , y1 , (x1 + w) , y1 , fill='#0FBA0F')
    canvas.create_line(x1 , y1 , x1 , (y1 - h) , fill='#0FBA0F')
    canvas.create_line(x2 , y2 , (x2 - w) , (y2) , fill='#0FBA0F')
    canvas.create_line(x2 , y2 , (x2) , (y2 + h) , fill='#0FBA0F')
    pred_label.config(text= 'Prediction: {} ({:.2f})'.format(np.argmax(val) , np.max(val)*100))
    canvas.pack()

def Clear():
    canvas.delete('all')
    global img
    img = np.zeros((100,100,1))


canvas.bind('<B1-Motion>' , Draw)
clear_button= tk.Button(root, text='Clear' , command= Clear)
predict_button= tk.Button(root , text='Predict' , command=lambda: predict(img))
l = tk.Label(root, text= "Double Click and Drag to draw.")
l.pack()
canvas.pack()
pred_label.pack()
clear_button.pack()
predict_button.pack()


root.mainloop()
