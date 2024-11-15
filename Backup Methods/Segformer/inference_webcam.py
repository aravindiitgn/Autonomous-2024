#Camera feed input

import cv2
import torch
import numpy as np
from torchvision import transforms as TF
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation
import time

# Load the trained model 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
model = SegformerForSemanticSegmentation.from_pretrained('nvidia/segformer-b2-finetuned-ade-512-512')

# Replace with the actual number of classes
model.config.num_labels = 2  

# Load the state from the fine-tuned model and set to model.eval() mode
model.load_state_dict(torch.load('Segformer/best_model.pth'))
model.to(device)
model.eval()

# Perform transformations
data_transforms = TF.Compose([
    TF.ToPILImage(),
    # TF.Resize((180, 320)),
    TF.Resize((360, 640)),
    TF.ToTensor(),
    TF.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Get camera input
cap = cv2.VideoCapture(0)

# Inference loop 
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
        start_time = time.time()  # Start time for FPS calculation
        
        # Preprocess the frame
        input_tensor = data_transforms(frame).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(pixel_values=input_tensor, return_dict=True)
            outputs = F.interpolate(outputs["logits"], size=(360, 640), mode="bilinear", align_corners=False)
            
            preds = torch.argmax(outputs, dim=1)
            preds = torch.unsqueeze(preds, dim=1)
            predicted_mask = (torch.sigmoid(preds) > 0.5).float()

        # Create an RGB version of the mask to overlay on the original frame
        mask_np = predicted_mask.cpu().squeeze().numpy()
        mask_resized = cv2.resize(mask_np, (frame.shape[1], frame.shape[0]))
        
        # Modify this section to create a green mask
        mask_rgb = np.zeros((mask_resized.shape[0], mask_resized.shape[1], 3), dtype=np.uint8)
        mask_rgb[:, :, 1] = (mask_resized * 255).astype(np.uint8)  # Set only the green channel

        # Post-processing for mask smoothening
        # Remove noise
        kernel = np.ones((3,3), np.uint8)
        opening = cv2.morphologyEx(mask_rgb, cv2.MORPH_OPEN, kernel, iterations=2)
        
        # Close small holes
        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=2)

        # Overlay the mask on the frame
        blended = cv2.addWeighted(frame, 0.65, closing, 0.6, 0)
        
        # Calculate FPS
        end_time = time.time()
        fps = 1.0 / (end_time - start_time)
        
        # Display FPS on the frame
        cv2.putText(blended, f"FPS: {fps:.2f}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Display the frame with the overlaid mask
        cv2.imshow('Lane Detection', blended)
        
        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
