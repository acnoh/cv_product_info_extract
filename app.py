import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from datetime import datetime
import tempfile
import easyocr
import os
import torch
from segment_anything import sam_model_registry, SamPredictor

yolo_model = YOLO('runs/train/product_model/weights/best.pt')
ocr_reader = easyocr.Reader(['en']) # ocr model

st.title("Key Product Features")
st.subheader("Annie Illing, Katherine Davis, and Andrea Noh")

uploaded_file = st.file_uploader("Upload an image", type=["jpeg", "jpg", "png"]) # upload image

if uploaded_file is not None:
    st.image(uploaded_file, caption='Uploaded image', use_container_width=True)
    
    # save the uploaded image to a temporary file (needed for YOLO and OpenCV)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpeg") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name

    results = yolo_model.predict(source=tmp_file_path, conf=0.5)
    quarter_diameter = 0.955  

    # extract bounding boxes and class indices
    bboxes = results[0].boxes.xyxy.cpu().numpy()
    classes = results[0].boxes.cls.cpu().numpy()
    quarter_bbox = None
    product_bbox = None

    # identify the quarter and product bounding boxes
    for bbox, cls in zip(bboxes, classes):
        label = yolo_model.names[int(cls)].lower()
        if label == "quarter":
            quarter_bbox = bbox
        elif label == "product":
            product_bbox = bbox

    if quarter_bbox is None or product_bbox is None:
        st.error("Both quarter and product bounding boxes must be detected.")
    else:
        # compute the quarter's pixel diameter (average of width and height)
        q_width = quarter_bbox[2] - quarter_bbox[0]
        q_height = quarter_bbox[3] - quarter_bbox[1]
        quarter_pixel_diameter = (q_width + q_height) / 2.0

        pixel_to_inch = quarter_diameter / quarter_pixel_diameter  # conversion factor: inches per pixel

        # get product dimensions in pixels and convert to inches
        p_width = product_bbox[2] - product_bbox[0]
        p_height = product_bbox[3] - product_bbox[1]
        product_width_in = p_width * pixel_to_inch
        product_height_in = p_height * pixel_to_inch

        st.success(f"Product dimensions: width = {product_width_in:.2f} inches, height = {product_height_in:.2f} inches")

        image_cv = cv2.imread(tmp_file_path)
        x1, y1, x2, y2 = map(int, product_bbox)
        product_crop = image_cv[y1:y2, x1:x2]

        ocr_results = ocr_reader.readtext(product_crop)
        product_size_text = None
        for bbox, text, conf in ocr_results:
            if "net wt." in text.lower() and conf >= 0.75:
                product_size_text = text
                break

        if product_size_text:
            st.success(f"Product Size: {product_size_text}")
        else:
            st.write("No product size information detected.")

        image_bgr = cv2.imread(tmp_file_path, cv2.IMREAD_COLOR) # original image
        HOME = os.getcwd()
        CHECKPOINT_PATH = os.path.join(HOME, "sam_vit_h_4b8939.pth")
        DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        MODEL_TYPE = "vit_h"
        sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH).to(DEVICE)
        mask_predictor = SamPredictor(sam)

        boxes_np = np.array([product_bbox])
        boxes_tensor = torch.from_numpy(boxes_np).to(torch.float)
        transformed_boxes = mask_predictor.transform.apply_boxes_torch(boxes_tensor, image_bgr.shape[:2])
        mask_predictor.set_image(image_bgr)

        # segmentation mask for the product
        masks, scores, logits = mask_predictor.predict_torch(
            boxes=transformed_boxes,
            multimask_output=False,  # Single mask prediction
            point_coords=None,
            point_labels=None
        )
        final_mask = masks[0][0].cpu().numpy().astype(np.uint8) * 255  # Scale mask for visualization

        # load and resize the promotional background image
        promo_img_path = "promo_dark.jpeg"
        promo_bgr = cv2.imread(promo_img_path, cv2.IMREAD_COLOR)
        if promo_bgr is None:
            st.error("Promo background image not found. Please ensure 'promo2.jpeg' exists in the working directory.")
        else:
            promo_resized = cv2.resize(promo_bgr, (640, 640))
            product_image = cv2.resize(image_bgr, (640, 640))
            final_mask_resized = cv2.resize(final_mask, (640, 640), interpolation=cv2.INTER_NEAREST)
            product_only = cv2.bitwise_and(product_image, product_image, mask=final_mask_resized) # product from the original image using the mask
            final_composite = promo_resized.copy()
            final_composite[final_mask_resized != 0] = product_only[final_mask_resized != 0] # product onto the promo background where the mask is non-zero

            final_composite_rgb = cv2.cvtColor(final_composite, cv2.COLOR_BGR2RGB) # convert final composite to RGB for display
            st.image(final_composite_rgb, caption="Product on Promo Background", use_container_width=True)

