import numpy as np
import cv2
import mediapipe as mp
import tensorflow as tf
import pickle
from collections import deque
from tensorflow.keras import layers # Ensure layers is imported
## used AI to help write these function

@tf.keras.utils.register_keras_serializable()
class TemporalAttention(layers.Layer):
    def __init__(self, units=64, **kwargs):
        super().__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        self.W = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer='glorot_uniform',
            trainable=True,
            name='attention_W'
        )
        self.b = self.add_weight(
            shape=(self.units,),
            initializer='zeros',
            trainable=True,
            name='attention_b'
        )
        self.u = self.add_weight(
            shape=(self.units,),
            initializer='glorot_uniform',
            trainable=True,
            name='attention_u'
        )
        super().build(input_shape)

    def call(self, x):
        # x shape: (batch, time, features)
        # Compute attention scores
        score = tf.nn.tanh(tf.tensordot(x, self.W, axes=1) + self.b)
        attention_weights = tf.nn.softmax(tf.tensordot(score, self.u, axes=1), axis=1)
        attention_weights = tf.expand_dims(attention_weights, -1)

        # Apply attention
        weighted = x * attention_weights
        return tf.reduce_sum(weighted, axis=1)

    def get_config(self):
        config = super().get_config()
        config.update({
            "units": self.units,
        })
        return config
    
class ImprovedFeatureExtractor:
    """Feature extractor matching the training pipeline"""
    
    def __init__(self):
        self.mp_holistic = mp.solutions.holistic
        
        # Key landmark indices
        self.key_face_indices = {
            'mouth': [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291,
                     308, 324, 318, 402, 317, 14, 87, 178, 88, 95],
            'eyes': [33, 160, 158, 133, 153, 144, 362, 385, 387, 263, 373, 380],
            'eyebrows': [70, 63, 105, 66, 107, 336, 296, 334, 293, 300],
            'nose': [1, 2, 98, 327, 6, 168]
        }
        
        self.key_pose_indices = [
            0, 11, 12, 13, 14, 15, 16, 23, 24
        ]
    
    def extract_geometric_features(self, results):
        """Extract hand-crafted geometric features"""
        features = []
        
        if results.face_landmarks:
            landmarks = results.face_landmarks.landmark
            
            # Mouth features
            upper_lip = landmarks[13]
            lower_lip = landmarks[14]
            mouth_left = landmarks[61]
            mouth_right = landmarks[291]
            
            mouth_height = np.sqrt((upper_lip.x - lower_lip.x)**2 +
                                  (upper_lip.y - lower_lip.y)**2 +
                                  (upper_lip.z - lower_lip.z)**2)
            mouth_width = np.sqrt((mouth_left.x - mouth_right.x)**2 +
                                 (mouth_left.y - mouth_right.y)**2 +
                                 (mouth_left.z - mouth_right.z)**2)
            
            mouth_aspect_ratio = mouth_height / (mouth_width + 1e-6)
            features.extend([mouth_height, mouth_width, mouth_aspect_ratio])
            
            # Tongue indicator
            mouth_center_x = (mouth_left.x + mouth_right.x) / 2
            mouth_center_y = (mouth_left.y + mouth_right.y) / 2
            tongue_indicator = np.sqrt((lower_lip.x - mouth_center_x)**2 +
                                      (lower_lip.y - mouth_center_y)**2)
            features.append(tongue_indicator)
            
            # Eye features
            left_eye_top = landmarks[159]
            left_eye_bottom = landmarks[145]
            right_eye_top = landmarks[386]
            right_eye_bottom = landmarks[374]
            
            left_eye_openness = np.sqrt((left_eye_top.x - left_eye_bottom.x)**2 +
                                       (left_eye_top.y - left_eye_bottom.y)**2)
            right_eye_openness = np.sqrt((right_eye_top.x - right_eye_bottom.x)**2 +
                                        (right_eye_top.y - right_eye_bottom.y)**2)
            
            features.extend([left_eye_openness, right_eye_openness])
            
            # Facial symmetry
            nose_tip = landmarks[1]
            face_symmetry = abs(nose_tip.x - 0.5)
            features.append(face_symmetry)
        else:
            features.extend([0.0] * 7)
        
        # Pose features
        if results.pose_landmarks:
            pose_landmarks = results.pose_landmarks.landmark
            
            left_shoulder = pose_landmarks[11]
            right_shoulder = pose_landmarks[12]
            left_wrist = pose_landmarks[15]
            right_wrist = pose_landmarks[16]
            
            left_arm_height = left_shoulder.y - left_wrist.y
            right_arm_height = right_shoulder.y - right_wrist.y
            
            features.extend([left_arm_height, right_arm_height])
            
            nose = pose_landmarks[0]
            hands_above_head = int(left_wrist.y < nose.y or right_wrist.y < nose.y)
            features.append(hands_above_head)
            
            shoulder_width = np.sqrt((left_shoulder.x - right_shoulder.x)**2 +
                                    (left_shoulder.y - right_shoulder.y)**2)
            features.append(shoulder_width)
        else:
            features.extend([0.0] * 4)
        
        # Hand features
        if results.left_hand_landmarks or results.right_hand_landmarks:
            hand_near_face = 0.0
            
            if results.left_hand_landmarks and results.face_landmarks:
                left_hand_center = np.mean([
                    [lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark
                ], axis=0)
                face_center = np.mean([
                    [landmarks[idx].x, landmarks[idx].y, landmarks[idx].z]
                    for idx in [1, 61, 291]
                ], axis=0)
                
                dist_to_face = np.linalg.norm(left_hand_center - face_center)
                hand_near_face = max(hand_near_face, 1.0 / (1.0 + dist_to_face * 5))
            
            if results.right_hand_landmarks and results.face_landmarks:
                right_hand_center = np.mean([
                    [lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark
                ], axis=0)
                face_center = np.mean([
                    [landmarks[idx].x, landmarks[idx].y, landmarks[idx].z]
                    for idx in [1, 61, 291]
                ], axis=0)
                
                dist_to_face = np.linalg.norm(right_hand_center - face_center)
                hand_near_face = max(hand_near_face, 1.0 / (1.0 + dist_to_face * 5))
            
            features.append(hand_near_face)
        else:
            features.append(0.0)
        
        return np.array(features)
    
    def extract_raw_landmarks(self, results):
        """Extract normalized raw landmark coordinates"""
        features = []
        
        # Face landmarks
        if results.face_landmarks:
            for category in ['mouth', 'eyes', 'eyebrows', 'nose']:
                for idx in self.key_face_indices[category]:
                    lm = results.face_landmarks.landmark[idx]
                    features.extend([lm.x, lm.y, lm.z])
        else:
            total_face_landmarks = sum(len(v) for v in self.key_face_indices.values())
            features.extend([0.0] * (total_face_landmarks * 3))
        
        # Pose landmarks
        if results.pose_landmarks:
            for idx in self.key_pose_indices:
                lm = results.pose_landmarks.landmark[idx]
                features.extend([lm.x, lm.y, lm.z, lm.visibility])
        else:
            features.extend([0.0] * (len(self.key_pose_indices) * 4))
        
        # Hand landmarks
        for hand_landmarks in [results.left_hand_landmarks, results.right_hand_landmarks]:
            if hand_landmarks:
                coords = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
                hand_center = np.mean(coords, axis=0)
                hand_spread = np.std(coords, axis=0)
                features.extend(hand_center.tolist())
                features.extend(hand_spread.tolist())
            else:
                features.extend([0.0] * 6)
        
        return np.array(features)
    
    def compute_temporal_features(self, sequence):
        """Compute velocity and acceleration features"""
        velocity = np.diff(sequence, axis=0, prepend=sequence[0:1])
        acceleration = np.diff(velocity, axis=0, prepend=velocity[0:1])
        enhanced = np.concatenate([sequence, velocity, acceleration], axis=1)
        return enhanced


def build_improved_model(input_shape, num_classes, dropout_rate=0.4):
    """
    Reconstructs the model architecture locally.
    Includes explicit output_shape definitions to prevent Lambda errors.
    """
    inputs = tf.keras.layers.Input(shape=input_shape)

    # Masking layer
    x = tf.keras.layers.Masking(mask_value=0.0)(inputs)

    # === Branch 1: Bidirectional LSTM with Attention ===
    lstm1 = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(128, return_sequences=True, dropout=0.3, recurrent_dropout=0.0)
    )(x)
    lstm1 = tf.keras.layers.LayerNormalization()(lstm1)

    lstm2 = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(64, return_sequences=True, dropout=0.3, recurrent_dropout=0.0)
    )(lstm1)
    lstm2 = tf.keras.layers.LayerNormalization()(lstm2)

    # Apply GLOBAL TemporalAttention
    attended = TemporalAttention(64)(lstm2)

    # === Branch 2: Temporal Convolutions ===
    # We strip the mask here to stop warnings, as Conv1D doesn't support it
    x_unmasked = tf.keras.layers.Lambda(lambda t: t)(x) 

    conv1 = tf.keras.layers.Conv1D(64, kernel_size=3, padding='same', activation='relu')(x_unmasked)
    conv1 = tf.keras.layers.BatchNormalization()(conv1)
    conv1 = tf.keras.layers.MaxPooling1D(pool_size=2)(conv1)

    conv2 = tf.keras.layers.Conv1D(64, kernel_size=5, padding='same', activation='relu')(x_unmasked)
    conv2 = tf.keras.layers.BatchNormalization()(conv2)
    conv2 = tf.keras.layers.MaxPooling1D(pool_size=2)(conv2)

    # Global pooling
    conv1_pooled = tf.keras.layers.GlobalAveragePooling1D()(conv1)
    conv2_pooled = tf.keras.layers.GlobalAveragePooling1D()(conv2)

    # === Branch 3: Statistical Features ===
    # Explicitly defining output_shape fixes the NotImplementedError
    def stat_output_shape(input_shape):
        return (input_shape[0], input_shape[2])

    stats_mean = tf.keras.layers.Lambda(lambda x: tf.reduce_mean(x, axis=1), output_shape=stat_output_shape)(x)
    stats_std = tf.keras.layers.Lambda(lambda x: tf.math.reduce_std(x, axis=1), output_shape=stat_output_shape)(x)
    stats_max = tf.keras.layers.Lambda(lambda x: tf.reduce_max(x, axis=1), output_shape=stat_output_shape)(x)

    # Merge all branches
    merged = tf.keras.layers.Concatenate()([
        attended, conv1_pooled, conv2_pooled, stats_mean, stats_std, stats_max
    ])

    # Dense layers
    dense1 = tf.keras.layers.Dense(256, activation='relu')(merged)
    dense1 = tf.keras.layers.BatchNormalization()(dense1)
    dense1 = tf.keras.layers.Dropout(dropout_rate)(dense1)

    dense2 = tf.keras.layers.Dense(128, activation='relu')(dense1)
    dense2 = tf.keras.layers.BatchNormalization()(dense2)
    dense2 = tf.keras.layers.Dropout(dropout_rate * 0.75)(dense2)

    # Residual connection
    dense2_residual = tf.keras.layers.Dense(128)(merged)
    dense2_combined = tf.keras.layers.Add()([dense2, dense2_residual])
    dense2_combined = tf.keras.layers.Activation('relu')(dense2_combined)

    dense3 = tf.keras.layers.Dense(64, activation='relu')(dense2_combined)
    dense3 = tf.keras.layers.Dropout(dropout_rate * 0.5)(dense3)

    # Output
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(dense3)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name='ImprovedEmoteClassifier')
    return model


# ================= REPLACE VideoGesturePredictor =================
class VideoGesturePredictor:
    def __init__(self, model_path='./models/video_model.keras', 
                 scaler_path='./models/feature_scaler.pkl',
                 sequence_length=30):
        
        print("Building model architecture locally...")
        
        self.model = build_improved_model(input_shape=(sequence_length, 621), num_classes=5)
        
        print(f"Loading weights from {model_path}...")
        # 2. Load ONLY the weights (Bypasses the 'Lambda' loading error)
        self.model.load_weights(model_path)
        
        print(f"Loading scaler from {scaler_path}...")
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
            
        self.sequence_length = sequence_length
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.extractor = ImprovedFeatureExtractor()
        self.frame_buffer = deque(maxlen=sequence_length)
        self.idx_to_class = {0: "Cry", 1: "HandsUp", 2: "Still", 
                            3: "TongueOut", 4: "Yawn"}
        
        print("✅ Model loaded successfully!")
    
    def extract_frame_features(self, frame):
        """Extract features from a single frame"""
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.holistic.process(image_rgb)
        
        
        geometric_features = self.extractor.extract_geometric_features(results)
        raw_features = self.extractor.extract_raw_landmarks(results)
        
        all_features = np.concatenate([geometric_features, raw_features])
        return all_features
    
    def predict_cr_video(self, frame):
        """
        Predict gesture from a single video frame
        Maintains internal buffer of recent frames for temporal prediction
        
        Args:
            frame: OpenCV BGR image (numpy array)
        
        Returns:
            prediction: Integer class index (0-4)
            probabilities: List of probabilities for each class
            class_name: String name of predicted class
        """
        # Extract features from current frame
        features = self.extract_frame_features(frame)
        
        self.frame_buffer.append(features)
        
        if len(self.frame_buffer) < self.sequence_length:
            return 2, [0.0, 0.0, 1.0, 0.0, 0.0], "Still"
        
        
        sequence = np.array(list(self.frame_buffer))  
        
        sequence_with_temporal = self.extractor.compute_temporal_features(sequence)
        
        sequence_normalized = self.scaler.transform(sequence_with_temporal)
        
        sequence_input = sequence_normalized.reshape(1, self.sequence_length, -1)
        
        # Predict
        predictions = self.model.predict(sequence_input, verbose=0)
        probabilities = predictions[0].tolist()
        predicted_class = np.argmax(predictions)
        class_name = self.idx_to_class[predicted_class]
        
        return predicted_class, probabilities, class_name
    
    def reset_buffer(self):
        """Clear the frame buffer (useful when switching videos or restarting)"""
        self.frame_buffer.clear()
    
    def __del__(self):
        """Cleanup MediaPipe resources"""
        if hasattr(self, 'holistic'):
            self.holistic.close()



_predictor = None

def predict_cr_video(frame):
    """
    Simple function interface for compatibility with existing code
    
    Args:
        frame: OpenCV BGR image (numpy array) - should be 244x244 or will be resized
    
    Returns:
        prediction: Integer class index (0-4)
    """
    global _predictor
    
    
    if _predictor is None:
        print("Initializing video gesture predictor...")
        _predictor = VideoGesturePredictor(
            model_path='./models/video_model.keras',
            scaler_path='./models/feature_scaler.pkl',
            sequence_length=30
        )
    
    # Get prediction
    pred_idx, probabilities, class_name = _predictor.predict_cr_video(frame)
    
    return pred_idx