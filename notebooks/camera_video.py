import cv2
import numpy as np
from PIL import Image
import time
from model_function_video import predict_cr_video


# --- CONFIGURATION CONSTANTS ---
WINDOW_WIDTH = 720
WINDOW_HEIGHT = 450
EMOJI_WINDOW_SIZE = (WINDOW_WIDTH, WINDOW_HEIGHT)

# Performance settings
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
GIF_FPS = 10  # Target FPS for GIF animation

# Class mappings
idx_to_class = {0:"Cry", 1:"HandsUp", 2:"Still", 3:"TongueOut", 4:"Yawn"}

# Helper function to load GIF frames with transparency support
def load_gif_frames(gif_path):
    gif = Image.open(gif_path)
    frames = []
    try:
        while True:
            # Convert to RGBA to preserve alpha channel
            frame = gif.convert('RGBA')
            
            # Create a white background
            background = Image.new('RGBA', frame.size, (255, 255, 255, 255))
            
            # Composite the frame onto the background
            composited = Image.alpha_composite(background, frame)
            
            # Convert to RGB then to OpenCV format
            composited_rgb = composited.convert('RGB')
            frame_cv = cv2.cvtColor(np.array(composited_rgb), cv2.COLOR_RGB2BGR)
            frame_resized = cv2.resize(frame_cv, EMOJI_WINDOW_SIZE)
            frames.append(frame_resized)
            
            gif.seek(gif.tell() + 1)
    except EOFError:
        pass
    return frames

# --- LOAD AND PREPARE IMAGES AND ANIMATIONS ---
try:
    CR_crying = load_gif_frames("./data/emotes/crying_goblin.gif")
    CR_confetti = load_gif_frames("./data/emotes/confetti-celebrate.gif")
    CR_67 = load_gif_frames("./data/emotes/67-meme-clash-royale.gif")
    CR_burning = load_gif_frames("./data/emotes/burning_goblin.gif")
    CR_toxic = load_gif_frames("./data/emotes/toxic-goblin.gif")
    CR_yawning = load_gif_frames("./data/emotes/yawning_princess.gif")

    print(f"All CR GIFs loaded successfully!")

except Exception as e:
    print("Error loading GIFs!")
    print(f"Error details: {e}")
    exit()

# Map predictions to animations
# 0: Cry, 1: HandsUp, 2: Still, 3: TongueOut, 4: Yawn
ANIMATION_MAP = {
    0: CR_crying,      # Cry
    1: CR_confetti,    # HandsUp
    2: CR_burning,     # Still
    3: CR_toxic,       # TongueOut
    4: CR_yawning,     # Yawn
}

# --- MAIN LOGIC ---
print("🎥 Searching for available cameras...")

# Find available cameras
available_cameras = []
for i in range(5):  # Check first 5 camera indices
    test_cap = cv2.VideoCapture(i)
    if test_cap.isOpened():
        print(f"✅ Camera {i} is available")
        available_cameras.append(i)
        test_cap.release()

if not available_cameras:
    print("❌ No cameras found!")
    exit()

print(f"\n📷 Using camera index: {available_cameras[0]}")
cap = cv2.VideoCapture(available_cameras[0])

cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
cap.set(cv2.CAP_PROP_FPS, 30)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Initialize windows
cv2.namedWindow('Camera Feed', cv2.WINDOW_NORMAL)
cv2.namedWindow('Animation Output', cv2.WINDOW_NORMAL)

cv2.resizeWindow('Camera Feed', WINDOW_WIDTH, WINDOW_HEIGHT)
cv2.resizeWindow('Animation Output', WINDOW_WIDTH, WINDOW_HEIGHT)

cv2.moveWindow('Camera Feed', 100, 100)
cv2.moveWindow('Animation Output', WINDOW_WIDTH + 150, 100)

print("Starting gesture detection...")
print("Press 'q' to quit")

# Animation tracking
current_animation = CR_burning  # Default animation (Still)
animation_frame_index = 0
last_gif_update = time.time()
gif_frame_delay = 1.0 / GIF_FPS

# Prediction tracking
last_prediction_time = time.time()
prediction_interval = 0.1  # Run prediction every 0.1 seconds (video model processes every frame internally)
current_gesture = "Still"  # Track current gesture name
buffer_filling = True  # Track if we're still filling the 30-frame buffer

frame_count = 0

while cap.isOpened():
    # READ CAMERA FRAME
    ret, frame = cap.read()
    
    if not ret:
        print("Failed to grab frame")
        break
    
    frame_count += 1
    
    # Resize camera frame to match window size for display
    camera_display = cv2.resize(frame, (WINDOW_WIDTH, WINDOW_HEIGHT))
    
    # RUN PREDICTION
    current_time = time.time()
    if current_time - last_prediction_time >= prediction_interval:
        # Call prediction function (it maintains its own internal buffer)
        prediction = predict_cr_video(frame)  # Returns 0-4
        
        # Check if buffer is still filling (first 30 frames)
        if buffer_filling and frame_count >= 30:
            buffer_filling = False
            print("Buffer filled! Predictions are now accurate.")
        
        if not buffer_filling:
            print(f"Prediction: {idx_to_class[prediction]} ({prediction})")
        
        # Update animation based on prediction
        if prediction in ANIMATION_MAP:
            new_gesture = idx_to_class[prediction]
            if new_gesture != current_gesture:  # Only update if gesture changed
                current_animation = ANIMATION_MAP[prediction]
                animation_frame_index = 0  # Reset to start of new animation
                current_gesture = new_gesture
                if not buffer_filling:
                    print(f"Detected: {current_gesture}")
        
        last_prediction_time = current_time
    
    # UPDATE ANIMATION FRAME
    if current_time - last_gif_update >= gif_frame_delay:
        animation_frame_index = (animation_frame_index + 1) % len(current_animation)
        last_gif_update = current_time
    
    # Get current animation frame
    animation_frame = current_animation[animation_frame_index]
    
    # Add gesture label to camera display
    status_text = "Gesture: Initializing..." if buffer_filling else f"Gesture: {current_gesture}"
    cv2.putText(camera_display, status_text, 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Add buffer status for first 30 frames
    if buffer_filling:
        cv2.putText(camera_display, f"Buffer: {frame_count}/30", 
                    (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)
    
    # DISPLAY BOTH WINDOWS
    cv2.imshow('Camera Feed', camera_display)
    cv2.imshow('Animation Output', animation_frame)
    
    # Exit on 'q' press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- CLEANUP ---
print("Shutting down...")
cap.release()
cv2.destroyAllWindows()
print("Application closed successfully!")