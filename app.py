import numpy as np
import os
import cv2
from tensorflow.keras.models import load_model
import streamlit as st
from streamlit_drawable_canvas import st_canvas

model = load_model('mnist.h5')
st.title('Handwritten Digit Recogniser')

menu = ['Home', 'Classifier', 'Creator']
choice = st.sidebar.selectbox('Menu', menu)

if choice == 'Home':
	st.header('About Project')
	st.write('Handwritten digit recognition is the ability of the computer to recognize human handwritten digits. This is a somewhat hard task for the machine because there is a myriad of different styles in which people write; in different shapes and curves')
	st.write('This model was trained using mnist Handwritten digit data; The mnist data contains 60,000 train set and 10,000 test set. Evaluation using these data partitions is limited, and so this application provides a GUI interface for user input evaluation')
	st.header('How it works:')
	st.write('1. From the menu section, navigate to the classifier option')
	st.write('2. In the GUI box provided, draw the digit you want the model to recognise')
	st.write('3. The model processes the image, and shows you what your input looks like after processing')
	st.write('4. The model recognises the digit you have entered, and constructs an accuracy graph to show you the accuracy level of the result')

	st.write('You can find the project source codes on [Github] (https://github.com/Taiwo-John/handwritten-digit-recognition.git)')


elif choice == 'Classifier': 
	st.header(''' Write a digit for the app to predict ''')
	width = 300
	height = width

	canvasRes = st_canvas(
		width=width,
		height=width,
		fill_color='#000000',
		stroke_width=15,
		stroke_color='#FFFFFF',
		background_color='#101526',
		key='canvas'
	)

	if canvasRes.image_data is not None: 
		img = cv2.resize(canvasRes.image_data.astype('float32'), (28, 28))
		img /= 255
		rescale = cv2.resize(img, (width, height), interpolation=cv2.INTER_NEAREST)
		st.header('Processed Model Input')
		st.image(rescale)

	if st.button('Recogise'):
		input_ = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		cval = model.predict(input_.reshape(1, 28, 28, 1))
		st.write(f'Your input is most likely: {np.argmax(cval[0])}')
		st.write('Hover on the graph bars for information on model accuracy value')
		st.bar_chart(cval[0])

else: 
	st.header('About Me')
	st.write('Taiwo is a creative techie who is passionate about fusing creativity and technology to build sustainable products.')
	st.write('He is interested in helping to build products in the fields of Finetch, Artificial Intelligence, and Blockchain.')
	st.write('Connect with Taiwo on [Linkedin!] (https://www.linkedin.com/in/taiwojohnt)')
	st.write('You can also send any feedback you have on the project to the above correspondence.')
	st.write('Thank You!')
	

