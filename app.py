import streamlit as st  
from ultralytics import YOLO
import random
import PIL
from PIL import Image, ImageOps
import numpy as np
import torchvision
import torch

from sidebar import Sidebar
import rcnnres, vgg
# hide deprication warnings which directly don't affect the working of the application
import warnings
warnings.filterwarnings("ignore")



# Sidebar 
sb = Sidebar()

title_image = sb.title_img
model = sb.model_name
conf_threshold = sb.confidence_threshold

#Main Page

st.title("Bone Fracture Detection")
st.write("The Application provides Bone Fracture Detection using multiple state-of-the-art computer vision models such as Yolo V8, ResNet, AlexNet, and CNN. To learn more about the app - Try it now!")

st.markdown("""
<style>

	.stTabs [data-baseweb="tab-list"] {
		gap: 10px;
    }

	.stTabs [data-baseweb="tab"] {
		height: 50px;
        white-space: pre-wrap;
		border-radius: 2px 2px 2px 2px;
		gap: 8px;
        padding-left: 10px;
        padding-right: 10px;
		padding-top: 8px;
		padding-bottom: 8px;
    }

	.stTabs [aria-selected="true"] {
  		background-color: #7f91ad;
	}

</style>""", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["Overview", "Test"])

with tab1:
   st.markdown("### Overview")
   st.text_area(
    "TEAM MEMBERS",
    "Ashita Shetty, Raj Motwani, Ramya Sri Gautham, Rishita Chebrolu, Akshita Agrawal, Revanth Chowdhary",
    )
   
   st.markdown("#### Network Architecture")
   network_img = "images\\NN_Architecture_Updated.jpg"
   st.image(network_img, caption="Network Architecture", width=None, use_column_width=True)
   
   st.markdown("#### Models Used")
   st.markdown("##### YoloV8")
   st.text_area(
       "Description",
       "In this bone fracture detection project, we're using a lightweight and efficient version of the YOLO v8 algorithm called YOLO v8 Nano (yolov8n). This algorithm is tailored for systems with limited computational resources. We start by training the model on a dataset of X-ray images that are labeled to show where fractures are. We specify the training settings in a YAML file. The model is trained for 50 epochs, and we save its progress every 25 epochs to keep the best-performing versions. YOLO v8 Nano is great at quickly and accurately spotting fractures, even on devices with lower computing power. After training, we test the model on a separate set of images to ensure it can reliably detect fractures. In practical use, the trained model automatically identifies and marks fractures on new X-ray images by drawing boxes around them. This helps doctors quickly and accurately diagnose fractures. We assess the model's effectiveness using performance metrics like confusion matrix and Intersection over Union (IoU) scores to understand how well it performs across different types of fractures.",
    )
   
   st.markdown("##### FasterRCNN with ResNet")
   st.text_area(
       "Decription",
       "This code implements a bone fracture detection system using the Fast R-CNN (Region-based Convolutional Neural Network) architecture. The purpose of Faster R-CNN (Region-based Convolutional Neural Network) is to perform efficient and accurate object detection within images. It addresses the challenge of localizing and classifying objects of interest in images, a fundamental task in computer vision applications. Faster R-CNN achieves this by introducing a Region Proposal Network (RPN) to generate candidate object bounding boxes, which are then refined and classified by subsequent network components. By combining region proposal generation and object detection into a single unified framework, Faster R-CNN significantly improves detection accuracy while maintaining computational efficiency, making it suitable for real-time applications such as autonomous driving, surveillance, medical imaging, and more. The dataset containing bone X-ray images is prepared, with images and their corresponding labels loaded and augmented to resize and transform boundary boxes. The Faster R-CNN model is then instantiated, with a pre-trained ResNet-50 backbone and a custom classification layer for bone fracture detection. The training loop is executed over multiple epochs, optimizing the model's parameters using the Adam optimizer and minimizing the combined loss. The best-performing model is saved, and its performance is evaluated on the validation set, with the best model further tested on a separate test set. After training, the model's ability to detect fractures is evaluated by comparing its predictions with the actual fractures. This helps us understand how accurate the model is in finding fractures. Also, a confusion matrix is created to see how well the model performs for different types of fractures, providing insights into its overall performance. By combining the power of ResNet's feature extraction capabilities with Faster R-CNN's precise object localization and classification, the system can effectively detect bone fractures within medical images with improved accuracy and reliability.",
       height=45,
    )
   
   st.markdown("##### SSD with VGG16")
   st.text_area(
       "Description",
       "This code uses a Single Shot Multibox Detector (SSD) with a VGG16 backbone to construct a bone fracture detection system. It performs preprocessing on training and validation datasets, doing augmentations including image scaling and bounding box coordinate conversion. Pre-trained weights are used to initialize the SSD300_VGG16 model, and additional custom layers are added to allow for fine-tuning to the particular purpose of fracture identification. The training loop is executed over multiple epochs, optimizing the model's parameters using the Adam optimizer and minimizing the combined loss. While the evaluation loop evaluates the model's performance on the validation dataset, the training loop continually runs over the dataset, calculating losses, back propagating, and adjusting weights using an optimizer. Overall, this code efficiently integrates SSD with VGG16 for real-time fracture detection, leveraging the model's ability to predict one of the 7 class labels directly from input images.",
    )
   
   
   
   
   
   
#weights 
yolo_path ="weights\yolov8.pt"

   
with tab2:
    st.markdown("### Upload & Test")
    #Image Uploading Button
    if 'clicked' not in st.session_state:
        st.session_state.clicked = False

    def set_clicked():
        st.session_state.clicked = True

    st.button('Upload Image', on_click=set_clicked)
    if st.session_state.clicked:
        image = st.file_uploader("", type=["jpg", "png"])
        
        
        if image is not None:
            st.write("You selected the file:", image.name)
            
            if model == 'YoloV8':
                try:
                    yolo_detection_model = YOLO(yolo_path)
                    yolo_detection_model.load()
                except Exception as ex:
                    st.error(f"Unable to load model. Check the specified path: {yolo_path}")
                    st.error(ex)
                
                col1, col2 = st.columns(2)

                with col1:
                    uploaded_image = PIL.Image.open(image)
                        
                    st.image(
                        image=image,
                        caption="Uploaded Image",
                        use_column_width=True
                    )

                    if uploaded_image:
                        if st.button("Execution"):
                            with st.spinner("Running..."):
                                res = yolo_detection_model.predict(uploaded_image,
                                                    conf=conf_threshold, augment=True, max_det=1)
                                boxes = res[0].boxes
                                res_plotted = res[0].plot()[:, :, ::-1]
                                
                                if len(boxes)==1:
                                
                                    names = yolo_detection_model.names
                                    probs = boxes.conf[0].item()
                                    
                                        
                                    for r in res:
                                        for c in r.boxes.cls:
                                            pred_class_label = names[int(c)]

                                    with col2:
                                        st.image(res_plotted,
                                                caption="Detected Image",
                                                use_column_width=True)
                                        try:
                                            with st.expander("Detection Results"):
                                                for box in boxes:
                                                    st.write(pred_class_label)
                                                    st.write(probs)
                                                    st.write(box.xywh)
                                        except Exception as ex:
                                            st.write("No image is uploaded yet!")
                                            st.write(ex)
                                
                                else:
                                    with col2:
                                        st.image(res_plotted,
                                                caption="Detected Image",
                                                use_column_width=True)
                                        try:
                                            with st.expander("Detection Results"):
                                                st.write("No Detection")
                                            #st.write(output[2])
                                        except Exception as ex:
                                            st.write("No Detection")
                                            st.write(ex)
                                    
                                        
            elif model == 'FastRCNN with ResNet':
                resnet_model = rcnnres.get_model()
                device = torch.device('cpu')
                resnet_model.to(device)

                
                col1, col2 = st.columns(2)

                with col1:
                    uploaded_image = PIL.Image.open(image)
                        
                    st.image(
                        image=image,
                        caption="Uploaded Image",
                        use_column_width=True
                    )
                    
                    content = Image.open(image).convert("RGB")
                    to_tensor = torchvision.transforms.ToTensor()
                    content = to_tensor(content).unsqueeze(0)
                    content.half()

                    if uploaded_image:
                        if st.button("Execution"):
                            with st.spinner("Running..."):
                                output = rcnnres.make_prediction(resnet_model, content, conf_threshold)
                                
                                print(output[0])

                                fig, _ax, class_name = rcnnres.plot_image_from_output(content[0].detach(), output[0])

                                with col2:
                                    st.image(rcnnres.figure_to_array(fig),
                                            caption="Detected Image",
                                            use_column_width=True)
                                    try:
                                        with st.expander("Detection Results"):
                                            st.write(class_name)
                                            st.write(output)
                                            #st.write(output[2])
                                    except Exception as ex:
                                        st.write("No image is uploaded yet!")
                                        st.write(ex)

            elif model == 'VGG16':
                vgg_model = vgg.get_vgg_model()
                device = torch.device('cpu')
                vgg_model.to(device)
                
                col1, col2 = st.columns(2)

                with col1:
                    uploaded_image = PIL.Image.open(image)
                        
                    st.image(
                        image=image,
                        caption="Uploaded Image",
                        use_column_width=True
                    )
                    
                    content = Image.open(image).convert("RGB")
                    to_tensor = torchvision.transforms.ToTensor()
                    content = to_tensor(content).unsqueeze(0)
                    content.half()

                    if uploaded_image:
                        if st.button("Execution"):
                            with st.spinner("Running..."):
                                output = rcnnres.make_prediction(vgg_model, content, conf_threshold)
                                
                                print(output[0])

                                fig, _ax, class_name = rcnnres.plot_image_from_output(content[0].detach(), output[0])

                                with col2:
                                    st.image(rcnnres.figure_to_array(fig),
                                            caption="Detected Image",
                                            use_column_width=True)
                                    try:
                                        with st.expander("Detection Results"):
                                            st.write(class_name)
                                            st.write(output)
                                            #st.write(output[2])
                                    except Exception as ex:
                                        st.write("No image is uploaded yet!")
                                        st.write(ex)

        
    else:
        st.write("Please upload an image to test")
            
        
        
            
            
            

            







        
        
        

        
        
        
        
