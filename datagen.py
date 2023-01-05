import numpy as np
import tensorflow as tf


def generate_dataset(X , Y , size=(20000, 250, 250 , 1) , min_max=(28 , 70)):
  new_X = []
  new_Y = [[] , []]
  dataset_len = size[0]
  for _ in range(dataset_len):
    bg = np.zeros((size[1] , size[2] , size[3]) , dtype='float32')
    idx = np.random.randint(0 , X.shape[0])
    new_Y[0].append(Y[idx])

    img = X[idx]
    height = width = np.random.randint(min_max[0] , min_max[1])
    img = tf.image.resize(img , (height , width) , method=tf.image.ResizeMethod.NEAREST_NEIGHBOR, preserve_aspect_ratio=True).numpy()
  
    x1 =  np.random.randint(0 , size[2] - width)
    y1 =  np.random.randint(0 , size[1] - height)
    x2 = x1 + width
    y2 = y1 + height
    bg[y1:y2 , x1:x2] = np.add(bg[y1:y2 , x1:x2] , img)

    x = (x1 + x2)/(2*size[2])
    y = (y1 + y2)/(2*size[1])
    h = (y2 - y1)/size[1]
    w = (x2 - x1)/size[2]

    labels = np.asarray([x,y,h,w] , dtype='float32')
    
    new_Y[1].append(labels)
    new_X.append(bg)
  
  new_X = np.asarray(new_X)
  new_Y[0] = np.asarray(new_Y[0])
  new_Y[1] = np.asarray(new_Y[1])
  return new_X , new_Y

if __name__ == '__main__':
  pass