import numpy as np
import cv2 as cv
import streamlit as st

def histogram(single_ch_img):
    count = []
    
    for color in range(256):
        sum_color = single_ch_img == color
        count.append(sum_color.sum())
    
    return np.array(count), np.arange(256)

original_img = cv.imread('lighting1.jpg')
gry_img = cv.imread('lighting1.jpg', 0)

b_img, g_img, r_img = cv.split(original_img)

# mask creation
# i would like more adjustment sliders, but the sidebar already look to crowded.
with st.sidebar:
    b_threshold = st.slider('blue_ch_thresh', 0, 256)
    g_threshold = st.slider('green_ch_thresh', 0, 256)
    r_threshold = st.slider('red_ch_thresh', 0, 256)
    
    addingB = st.slider('blue_adjustment', 0, 256)
    addingG = st.slider('green_adjustment', 0, 256)
    addingR = st.slider('red_adjustment', -100, 256, 0) # having the range be negative will allow for substraction as well as addition.

# the thresholding is fine, but i will add the ability to use differnt threshold methods later.
_, b_mask = cv.threshold(b_img, b_threshold, 255, cv.THRESH_BINARY)
_, g_mask = cv.threshold(g_img, g_threshold, 255, cv.THRESH_BINARY)
_, r_mask = cv.threshold(r_img, r_threshold, 255, cv.THRESH_BINARY)

#this shows my bgr channel masks
col_mask1, col_mask2, col_mask3 = st.columns(3)

with col_mask1:
    st.image(b_mask, caption='blue_ch_thresh')
with col_mask2:
    st.image(g_mask, caption='green_ch_thresh')
with col_mask3:
    st.image(r_mask, caption='red_ch_thresh')

#histograms of the original image channels. 
b_count, b_color = histogram(b_img)
g_count, g_color = histogram(g_img)
r_count, r_color = histogram(r_img)

hist_display = st.multiselect('Histograms', ['blueHist', 'greenHist', 'redHist'])

# might put this above the masks
with st.expander('histograms graphs'):
    if 'blueHist' in hist_display:
        st.bar_chart(b_count)
    if 'greenHist' in hist_display:
        st.bar_chart(g_count)
    if 'redHist' in hist_display:
        st.bar_chart(r_count)
        
# image displays
b_adjustment = cv.add(b_img, addingB, mask=b_mask)
g_adjustment = cv.add(g_img, addingG, mask=g_mask)
r_adjustment = cv.add(r_img, addingR, mask=r_mask)
bgr_adjustment = cv.merge((b_adjustment, g_adjustment, r_adjustment))

col1, col2 = st.columns(2)
    
with col1:
    st.image(original_img, channels='BGR')
with col2:
    st.image(bgr_adjustment, channels='BGR') 

st.cache(histogram)
