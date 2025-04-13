import cv2
import os
from collections import defaultdict

# Create directories
os.makedirs("sign_images", exist_ok=True)

# Sign database (matches your recognition system)
SIGNS_TO_CAPTURE = [
    "hello", "thank you", "yes", "no", "please", 
    "i love you", "food", "water", "sorry",
    "good morning", "good night", "friends",
    "help", "happy", "sad", "angry",
    "tired", "scared", "excited", "bored"
]

# Webcam setup
cap = cv2.VideoCapture(0)
current_sign = None
image_count = defaultdict(int)
samples_per_sign = 50  # Number of images to capture per sign

def show_instructions(frame):
    instructions = [
        "Sign Image Capture",
        "Press:",
        "0-9: Select sign (see console)",
        "Space: Capture image",
        "q: Quit"
    ]
    
    y_offset = 30
    for line in instructions:
        cv2.putText(frame, line, (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        y_offset += 30
    
    if current_sign:
        status = f"Capturing: {current_sign} [{image_count[current_sign]}/{samples_per_sign}]"
        cv2.putText(frame, status, (10, y_offset+30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

# Print sign menu
print("Select sign to capture:")
for i, sign in enumerate(SIGNS_TO_CAPTURE[:10]):
    print(f"{i}: {sign}")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.flip(frame, 1)
    show_instructions(frame)
    
    cv2.imshow("Sign Capture", frame)
    
    key = cv2.waitKey(1) & 0xFF
    
    # Select sign (0-9)
    if 48 <= key <= 57:  # ASCII for 0-9
        sign_index = key - 48
        if sign_index < len(SIGNS_TO_CAPTURE):
            current_sign = SIGNS_TO_CAPTURE[sign_index]
            print(f"Selected: {current_sign}")
    
    # Capture image (spacebar)
    elif key == 32 and current_sign:
        if image_count[current_sign] < samples_per_sign:
            # Create clean filename
            filename = current_sign.replace(" ", "_") + ".png"
            filepath = os.path.join("sign_images", filename)
            
            # Save image
            cv2.imwrite(filepath, frame)
            image_count[current_sign] += 1
            print(f"Saved {filepath} ({image_count[current_sign]}/{samples_per_sign})")
        else:
            print(f"Already captured {samples_per_sign} samples for {current_sign}")
    
    # Quit (q)
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Image capture complete!")