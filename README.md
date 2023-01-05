
# Object Localization Using Mnist

Object localization using bounding box regression technique in Keras and 
interactively visualizing the model’s prediction using a GUI.



## Object Localization

Object Localization is a technique where you identify a particular instance of an object
in an image and locate it, typically by specifying a tightly cropped bounding box centered on the instance.

To learn more about it check out [Andrew Ng's lecture](https://www.youtube.com/watch?v=GSwYGkTfOKk) of Object localization.
![](https://miro.medium.com/max/1100/0*sW8lbS9CwbyE5cXm.webp)


## The Model
I have built an object classification and localization model
which is trained on a synthetic dataset.
![](https://miro.medium.com/max/1100/0*-7pdw0pSFbwNJmpL.webp)

### Localization

The localization is done using bounding box regression method.
The model returns an output(regression head) with 4 numbers (x , y , h , w).

(x , y) is the central coordinate of the object and (h , w) are 
ratios of height of box to image and width of box to image respectively.

## The Dataset

I am using a synthetic dataset which I have generated
using the [datagen.py](https://github.com/Parijat-18/Mnist-Object-localization/blob/main/datagen.py). 
![](https://raw.githubusercontent.com/Parijat-18/Mnist-Object-localization/main/sample.png)

I have an [example notebook](https://github.com/Parijat-18/Mnist-Object-localization/blob/main/example.ipynb) 
where you can take a look at the implementation
of preprocessing, generating and training the dataset.


## GUI using Tkinter for visualizing the model.

I have added an easy python script to see your model 
predict classes and draw bounding boxes on a canvas
using Tkinter. You can check out the [source code](https://github.com/Parijat-18/Mnist-Object-localization/blob/main/gui_digitPred.py).
## Installation

Easy installation guide

```bash
    C:\Users\parij>pip install tk
    C:\Users\parij>git clone https://github.com/Parijat-18/Mnist-Object-localization.git
    C:\Users\parij>cd Mnist-Object-localization
    C:\Users\parij\Mnist-Object-localization>python gui_digitPred.py
```
    
## Acknowledgements

 - [Andrew Ng](https://www.youtube.com/watch?v=GSwYGkTfOKk)
 - [Object Localization with Keras and W&B](https://medium.com/analytics-vidhya/object-localization-with-keras-2f272f79e03c)

