import numpy as np
import cv2
import dlib
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image, ImageColor

# 랜드마크 가져오기 - 여러명인 경우 보완 필요
def getLandmarks(faceDetector, landmarkDetector, image) :
    FRect = faceDetector(image, 0)[0]
    Area = FRect.area()
    rect = dlib.rectangle(FRect.left(), FRect.top(), FRect.right(), FRect.bottom())
   
    landmarks = landmarkDetector(image, rect)
    points = []
    for p in landmarks.parts():
        pt = (p.x, p.y)
        points.append(pt)
    
    return points

# 컬러 필터 - BGR -> RGB 
# HSV에서 변환이라 어두운 색 인식 잘 못할 가능성 - 나중에 생각
def change_color(image, color=[168, 68, 90]):
    b, g, r = color 
    filter_color = np.zeros_like(image)
    filter_color[:, :, 0:3] = [b, g, r]

    img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    filter_hsv = cv2.cvtColor(filter_color, cv2.COLOR_BGR2HSV)

    img_hsv[:, :, 0:2] = filter_hsv[:, :, 0:2]
    
    filtered_img = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)
    return filtered_img

# 이미지 처리 - 마스킹, 합체
def image_process(im, lip_color):
    faceDetector = dlib.get_frontal_face_detector()
    landmarkDetector = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    landmarks = getLandmarks(faceDetector, landmarkDetector, im)

    mask = np.zeros_like(im)

    # 입술 랜드 마크 (입술 외부 = [48:60], 입술 내부 = [60:68])
    lipPoints_out = landmarks[48:60]
    lipPoints_in = landmarks[60:68]

    # 입술 마스킹 처리
    cv2.fillPoly(mask, [np.array(lipPoints_out)], (255, 255, 255))
    cv2.fillPoly(mask,[np.array(lipPoints_in)], (0, 0, 0))

    # 마스크 블러처리
    mask = cv2.medianBlur(mask, 7)
    mask = cv2.GaussianBlur(mask,(11,11),10)

    # 입술제외 마스킹
    inverse_mask = cv2. bitwise_not(mask)

    # 입술 색깔 필터 씌우기 
    mask_color = change_color(im, lip_color)

    # 정규화
    mask = mask.astype(float)/255
    inverse_mask = inverse_mask.astype(float)/255
    mask_color = mask_color.astype(float)/255
    face = im.astype(float)/255

    # 입술, 얼굴만 각각 추출
    lips_only = cv2.multiply(mask, mask_color)
    face_only = cv2.multiply(inverse_mask, face)

    # 합치기
    result = lips_only + face_only

    return result

# 이미지 처리 - 크기, 비율 // 현재 미사용
def resizeImg_tosquare(img, img_size) :
    
    try :
        h,w,c = img.shape
    except :
        print("이미지 다시 확인")

    if h < w :
        new_w = img_size
        new_h = int((h/w)*img_size)
    else :
        new_h = img_size
        new_w = int((w/h)*img_size)

    cv2.resize(img, (new_w, new_h), cv2.INTER_AREA)
    pad_w = (img_size - new_w)//2
    pad_h = (img_size - new_h)//2

    resize_img = cv2.copyMakeBorder(img, pad_w,pad_w,pad_h,pad_h,cv2.BORDER_CONSTANT, value=[0,0,0])

    return resize_img

# Streamlit 설정
def main() :
    st.title('Virtual Makeup (Lips) Demo')
    st.sidebar.title('Virtual Makeup Settings')

    DEMO_IMG = "imgs\DEMO_img.jpg"
    DEMO_IMG1 = 'imgs\DEMO_img1.jpg'
    DEMO_IMG2 = 'imgs\DEMO_img2.jpg'
    DEMO_IMG3 = 'imgs\DEMO_ASIAN.jpg'
    DEMO_IMG4 = 'imgs\DEMO_ASIAN1.jpg'

    img_file_buffer = st.sidebar.file_uploader("파일 업로드", type=[ "jpg", "jpeg",'png'])

    if img_file_buffer is not None:
        im = np.array(Image.open(img_file_buffer))
        demo_image = img_file_buffer

    else:
        demo_image = DEMO_IMG
        im = np.array(Image.open(demo_image))

    DEMO_IMG1_or = np.array(Image.open(DEMO_IMG1))
    DEMO_IMG2_or = np.array(Image.open(DEMO_IMG2))
    DEMO_IMG3_or = np.array(Image.open(DEMO_IMG3))
    DEMO_IMG4_or = np.array(Image.open(DEMO_IMG4))

    # 립컬러 선택
    lip_color = st.sidebar.color_picker('립스틱 색상', '#A8445A')
    lip_color = ImageColor.getcolor(lip_color, "RGB")

    image = image_process(im, lip_color)


    DEMO_IMG1_ch = image_process(DEMO_IMG1_or, lip_color)
    DEMO_IMG2_ch = image_process(DEMO_IMG2_or, lip_color)
    DEMO_IMG3_ch = image_process(DEMO_IMG3_or, lip_color)
    DEMO_IMG4_ch = image_process(DEMO_IMG4_or, lip_color)


    tab1, tab2, tab3 = st.tabs(['Your Choice', 'Dark Skin', 'Bright Skin'])
    with tab1 :
        col1_1, col1_2 = st.columns(2)
        with col1_1 :    
            st.subheader('Original Image')
            st.image(im,use_column_width = True)
        with col1_2 :
            st.subheader('Output Image')
            st.image(image,use_column_width = True)

    with tab2 :
        col2_1, col2_2, col2_3, col2_4 = st.columns(4)
        with col2_1 :    
            st.subheader('Input')
            st.image(DEMO_IMG1_or,use_column_width = True)
        with col2_2 :
            st.subheader('Output')
            st.image(DEMO_IMG1_ch,use_column_width = True)
        with col2_3:
            st.subheader("Input")
            st.image(DEMO_IMG2_or,use_column_width = True)
        with col2_4:
            st.subheader('Output')
            st.image(DEMO_IMG2_ch,use_column_width = True)

    with tab3 :
        col3_1, col3_2, col3_3, col3_4 = st.columns(4)
        with col3_1 :    
            st.subheader('Input')
            st.image(DEMO_IMG3_or,use_column_width = True)
        with col3_2 :
            st.subheader('Output')
            st.image(DEMO_IMG3_ch,use_column_width = True)
        with col3_3:
            st.subheader("Input")
            st.image(DEMO_IMG4_or,use_column_width = True)
        with col3_4:
            st.subheader('Output')
            st.image(DEMO_IMG4_ch,use_column_width = True)

    st.text('')

if __name__ == '__main__':
    main()
  
