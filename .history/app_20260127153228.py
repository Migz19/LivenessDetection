import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image
import os
from pathlib import Path
import tempfile
from datetime import datetime

# Import custom modules
from models.cnn_model import load_cnn_model
from models.efficientnet_model import load_efficientnet_model
from utils.preprocessing import ImagePreprocessor, VideoPreprocessor
from utils.face_detection import FaceDetector, MultiiFaceProcessor
from utils.inference import LivenessInference
from utils.enhanced_inference import EnhancedLivenessInference

# Set page config
st.set_page_config(
    page_title="Liveness Detection App",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding-top: 2rem;
    }
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 1.2rem;
        font-weight: bold;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 1rem;
        border-radius: 0.25rem;
    }
    .error-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        padding: 1rem;
        border-radius: 0.25rem;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_models(model_type: str):
    """Load models based on selection"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if model_type == "CNN":
        model = load_cnn_model(device=device)
    else:  # EfficientNet
        model = load_efficientnet_model(device=device, pretrained=True)
    
    return model, device


@st.cache_resource
def get_face_detector():
    """Get face detector"""
    return FaceDetector(), MultiiFaceProcessor()


def display_detection_results(prediction: str, confidence: float, faces_count: int = 1, temporal_info: dict = None):
    """Display detection results in a nice format"""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Prediction", prediction)
    
    with col2:
        st.metric("Confidence", f"{confidence:.2%}")
    
    with col3:
        st.metric("Faces Detected", faces_count)
    
    # Color coded result
    if prediction == "Live":
        st.success(f"✅ **LIVE FACE DETECTED** - Confidence: {confidence:.2%}")
    else:
        st.error(f"❌ **FAKE/SPOOF DETECTED** - Confidence: {confidence:.2%}")
    
    # Show temporal smoothing indicator
    if temporal_info and temporal_info.get('smoothed_confidence') is not None:
        st.divider()
        col1, col2 = st.columns(2)
        
        with col1:
            original_conf = temporal_info.get('original_confidence', confidence)
            st.write(f"**Raw Prediction Confidence:** {original_conf:.2%}")
        
        with col2:
            st.write(f"**After Transformer:** {temporal_info.get('smoothed_confidence'):.2%}")
        
        # Calculate change
        change = temporal_info.get('smoothed_confidence', 0) - temporal_info.get('original_confidence', confidence)
        if abs(change) > 0.01:
            if change > 0:
                st.info(f"🔄 **Transformer increased confidence by {abs(change):.2%}** (temporal smoothing applied)")
            else:
                st.warning(f"🔄 **Transformer decreased confidence by {abs(change):.2%}** (temporal smoothing applied)")
        else:
            st.info(f"🔄 **Transformer applied** (minimal change: {change:+.2%})")


def process_image_input(image_input, model, device, preprocessor, face_detector, model_type):
    """Process image input and perform liveness detection"""
    
    # Convert image to numpy array
    if isinstance(image_input, Image.Image):
        image_array = np.array(image_input)
        if len(image_array.shape) == 3 and image_array.shape[2] == 3:
            image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
    else:
        image_array = image_input
    
    # Detect faces
    faces = face_detector.detect_faces(image_array)
    
    if len(faces) == 0:
        st.warning("⚠️ No faces detected in the image!")
        return None
    
    # Display detected faces
    st.info(f"Detected {len(faces)} face(s)")
    
    # Process each face with enhanced inference
    inference = EnhancedLivenessInference(model, device)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Original Image")
        image_with_boxes = face_detector.draw_faces(image_array, faces)
        st.image(cv2.cvtColor(image_with_boxes, cv2.COLOR_BGR2RGB))
    
    with col2:
        st.subheader("Detection Results")
        
        all_predictions = []
        all_confidences = []
        
        for idx, face in enumerate(faces):
            bbox = face['bbox']
            result = inference.predict_single_with_features(image_array, bbox)
            pred = result['prediction']
            conf = result['adjusted_confidence']
            all_predictions.append(pred)
            all_confidences.append(conf)
            
            # Display face crop
            face_crop = face_detector.crop_face(image_array, bbox)
            
            st.write(f"**Face {idx + 1}:**")
            st.image(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB), use_column_width=True)
            
            if pred == "Live":
                st.success(f"✅ Live - Confidence: {conf:.2%}")
            else:
                st.error(f"❌ Fake - Confidence: {conf:.2%}")
        
        # Overall summary
        st.divider()
        live_count = sum(1 for p in all_predictions if p == "Live")
        overall_pred = "Live" if live_count > len(faces) / 2 else "Fake"
        overall_conf = max(
            sum(c for p, c in zip(all_predictions, all_confidences) if p == "Live") / max(live_count, 1),
            sum(c for p, c in zip(all_predictions, all_confidences) if p == "Fake") / max(len(faces) - live_count, 1)
        )
        
        display_detection_results(overall_pred, overall_conf, len(faces))
        
        return {
            'predictions': all_predictions,
            'confidences': all_confidences,
            'overall': overall_pred
        }


def process_video_input(video_path, model, device, preprocessor, face_detector, 
                       video_preprocessor, model_type, num_frames=10):
    """Process video and perform liveness detection"""
    
    # Extract frames
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    status_text.text("Extracting frames...")
    frames = video_preprocessor.extract_frames(video_path, num_frames=num_frames)
    
    if len(frames) == 0:
        st.error("Could not extract frames from video!")
        return None
    
    progress_bar.progress(30)
    
    # Detect faces in first frame to determine bboxes
    status_text.text("Detecting faces...")
    faces = face_detector.detect_faces(frames[0])
    
    if len(faces) == 0:
        st.warning("⚠️ No faces detected in video!")
        return None
    
    # Use the primary (first) detected face's bbox for all frames
    # This assumes tracking of the same face throughout video
    face_bboxes = [faces[0]['bbox']]  # Single bbox for all frames
    
    progress_bar.progress(50)
    
    # Run inference with motion analysis
    status_text.text("Running liveness detection...")
    inference = EnhancedLivenessInference(model, device)
    results = inference.predict_video_with_motion(frames, face_bboxes)
    
    progress_bar.progress(100)
    status_text.empty()
    progress_bar.empty()
    
    # Display results
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Video Analysis")
        st.write(f"**Total Frames Analyzed:** {len(frames)}")
        st.write(f"**Faces Detected:** {len(faces)}")
        st.write(f"**Live Frames:** {results['live_count']}")
        st.write(f"**Fake Frames:** {results['fake_count']}")
    
    with col2:
        st.subheader("Frame-by-Frame Results")
        for idx, (pred, conf) in enumerate(zip(results['predictions'], results['confidences'])):
            status = "✅ Live" if pred == "Live" else "❌ Fake"
            st.write(f"Frame {idx + 1}: {status} ({conf:.2%})")
    
    # Overall result
    st.divider()
    st.subheader("Overall Video Result")
    
    # Prepare temporal info for display
    temporal_info = {
        'smoothed_confidence': results.get('smoothed_confidence') or results.get('final_confidence'),
        'original_confidence': results.get('original_confidence', np.mean(results.get('confidences', [0])))
    }
    
    display_detection_results(
        results['final_prediction'],
        results['final_confidence'],
        len(faces),
        temporal_info=temporal_info
    )
    
    # Show sample frames with detections
    st.subheader("Sample Frames with Face Detection")
    sample_indices = np.linspace(0, len(frames) - 1, min(3, len(frames)), dtype=int)
    
    cols = st.columns(len(sample_indices))
    for idx, col in zip(sample_indices, cols):
        with col:
            frame_with_boxes = face_detector.draw_faces(
                frames[idx], faces,
                confidences=[results['confidences'][idx]] * len(faces)
            )
            st.image(cv2.cvtColor(frame_with_boxes, cv2.COLOR_BGR2RGB))
            st.caption(f"Frame {idx + 1}: {results['predictions'][idx]}")
    
    return results


def main():
    """Main Streamlit application"""
    
    st.title("🎥 Liveness Detection Application")
    st.markdown("---")
    
    # Sidebar configuration
    st.sidebar.title("⚙️ Configuration")
    
    model_type = st.sidebar.radio(
        "Select Model",
        ["CNN", "EfficientNet"],
        help="CNN: Lightweight model, EfficientNet: More robust pre-trained model"
    )
    
    device_info = st.sidebar.info(
        f"🖥️ **Device:** {'CUDA (GPU)' if torch.cuda.is_available() else 'CPU'}"
    )
    
    # Load models
    st.sidebar.subheader("Loading Models...")
    model, device = load_models(model_type)
    face_detector, multi_face_processor = get_face_detector()
    
    # Create preprocessors
    if model_type == "CNN":
        preprocessor = ImagePreprocessor(model_type='cnn')
    else:
        preprocessor = ImagePreprocessor(model_type='efficientnet')
    
    video_preprocessor = VideoPreprocessor(model_type=model_type.lower())
    
    st.sidebar.success(f"✅ {model_type} model loaded successfully!")
    
    # Main tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📷 Image Detection",
        "🎬 Video Detection",
        "📹 Webcam Detection",
        "📊 Batch Processing",
        "ℹ️ About"
    ])
    
    # ============== TAB 1: IMAGE DETECTION ==============
    with tab1:
        st.header("📷 Image Liveness Detection")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Upload Image")
            image_file = st.file_uploader(
                "Choose an image file",
                type=["jpg", "jpeg", "png", "bmp"],
                help="Upload an image containing faces"
            )
            
            if image_file:
                image = Image.open(image_file)
                st.image(image, caption="Uploaded Image", use_column_width=True)
                
                if st.button("🔍 Detect Liveness", key="image_detect"):
                    with st.spinner("Processing..."):
                        results = process_image_input(
                            image, model, device, preprocessor,
                            face_detector, model_type
                        )
        
        with col2:
            st.subheader("Instructions")
            st.markdown("""
            1. Upload a clear image with faces
            2. The app will detect all faces in the image
            3. Each face will be analyzed for liveness
            4. Results show:
               - ✅ **Live**: Real face detected
               - ❌ **Fake**: Spoof/fake face detected
               - Confidence score for each prediction
            
            **Tips for best results:**
            - Ensure good lighting
            - Keep faces clearly visible
            - Avoid blurry images
            """)
    
    # ============== TAB 2: VIDEO DETECTION ==============
    with tab2:
        st.header("🎬 Video Liveness Detection")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Upload Video")
            video_file = st.file_uploader(
                "Choose a video file",
                type=["mp4", "avi", "mov", "mkv"],
                help="Upload a video for liveness detection"
            )
            
            if video_file:
                # Save video temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
                    tmp_file.write(video_file.read())
                    tmp_path = tmp_file.name
                
                # Video preview
                st.video(tmp_path)
                
                # Frame selection
                num_frames = st.slider(
                    "Number of frames to analyze",
                    min_value=5,
                    max_value=30,
                    value=10,
                    help="More frames = more thorough but slower analysis"
                )
                
                if st.button("🔍 Detect Liveness", key="video_detect"):
                    with st.spinner("Analyzing video..."):
                        results = process_video_input(
                            tmp_path, model, device, preprocessor,
                            face_detector, video_preprocessor,
                            model_type, num_frames=num_frames
                        )
                
                # Cleanup
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
        
        with col2:
            st.subheader("Instructions")
            st.markdown("""
            1. Upload a video file
            2. Select number of frames to analyze
            3. The app will:
               - Extract frames from the video
               - Detect faces in each frame
               - Analyze each face for liveness
               - Aggregate results
            
            **Interpretation:**
            - More "Live" frames → Likely real face
            - More "Fake" frames → Likely spoofed face
            - Consistency → Higher confidence
            """)
    
    # ============== TAB 3: WEBCAM DETECTION ==============
    with tab3:
        st.header("📹 Webcam Liveness Detection")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Webcam Input")
            
            num_frames = st.slider(
                "Frames to capture (video duration)",
                min_value=5,
                max_value=30,
                value=15,
                key="webcam_frame_count"
            )
            
            col_btn1, col_btn2 = st.columns(2)
            with col_btn1:
                if st.button("📹 Capture Video from Webcam", key="webcam_capture"):
                    with st.spinner(f"Recording {num_frames} frames from webcam..."):
                        frames = video_preprocessor.extract_frames_from_webcam(num_frames=num_frames)
                        
                        if len(frames) > 0:
                            # Convert frames to bytes for serialization in session state
                            frames_bytes = []
                            for frame in frames:
                                _, buffer = cv2.imencode('.jpg', frame)
                                frames_bytes.append(buffer.tobytes())
                            
                            # Store serialized frames in session state (use different key name to avoid conflict with slider)
                            st.session_state.captured_webcam_frames = frames_bytes
                            st.session_state.webcam_frames_captured = True
                            st.success(f"✅ Captured {len(frames)} frames")
                            st.rerun()
                        else:
                            st.error("Could not access webcam!")
            
            with col_btn2:
                if st.session_state.get('webcam_frames_captured', False):
                    if st.button("🔄 Clear Recording", key="clear_webcam"):
                        st.session_state.captured_webcam_frames = None
                        st.session_state.webcam_frames_captured = False
                        st.rerun()
            
            # Display captured frames and analysis button if frames exist
            if st.session_state.get('webcam_frames_captured', False) and st.session_state.get('captured_webcam_frames'):
                # Convert bytes back to numpy arrays
                frames_bytes = st.session_state.captured_webcam_frames
                frames = []
                for frame_bytes in frames_bytes:
                    nparr = np.frombuffer(frame_bytes, np.uint8)
                    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    frames.append(frame)
                
                st.info(f"📹 **Recorded {len(frames)} frames** (click Analyze to process)")
                
                # Show video preview with frame slider
                st.subheader("Video Preview")
                frame_idx = st.slider(
                    "Frame",
                    min_value=0,
                    max_value=len(frames)-1,
                    value=0,
                    key="webcam_frame_slider"
                )
                
                st.image(cv2.cvtColor(frames[frame_idx], cv2.COLOR_BGR2RGB), 
                        use_column_width=True,
                        caption=f"Frame {frame_idx + 1}/{len(frames)}")
                
                # Show frame grid thumbnails
                with st.expander("👁️ View All Frames as Grid"):
                    cols = st.columns(4)
                    for idx, frame in enumerate(frames):
                        with cols[idx % 4]:
                            st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), 
                                    use_column_width=True,
                                    caption=f"#{idx+1}")
                
                st.divider()
                
                # Run detection
                if st.button("🔍 Analyze Captured Video", key="analyze_webcam"):
                    with st.spinner("Running liveness detection with temporal transformer..."):
                        # Detect faces in first frame
                        faces = face_detector.detect_faces(frames[0])
                        
                        if len(faces) > 0:
                            face_bboxes = [faces[0]['bbox']]  # Use primary face only
                            
                            # Run inference with motion analysis (temporal smoothing enabled by default)
                            inference = EnhancedLivenessInference(model, device, use_temporal_smoothing=True)
                            results = inference.predict_video_with_motion(frames, face_bboxes)
                            
                            st.divider()
                            
                            # Prepare temporal info for display
                            temporal_info = {
                                'smoothed_confidence': results.get('smoothed_confidence') or results.get('final_confidence'),
                                'original_confidence': results.get('original_confidence', np.mean(results.get('confidences', [0])))
                            }
                            
                            display_detection_results(
                                results['final_prediction'],
                                results['final_confidence'],
                                len(faces),
                                temporal_info=temporal_info
                            )
                            
                            # Show metrics
                            st.subheader("Frame Analysis")
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.write(f"**Live Frames:** {results['live_count']}/{len(frames)}")
                            with col2:
                                st.write(f"**Fake Frames:** {results['fake_count']}/{len(frames)}")
                            with col3:
                                st.write(f"**Motion Prediction:** {results['motion_prediction']}")
                        else:
                            st.warning("No faces detected in captured frames!")
        
        with col2:
            st.subheader("Instructions")
            st.markdown("""
            1. Click "Capture from Webcam"
            2. Allow camera access when prompted
            3. The app will capture frames
            4. Click "Analyze Captured Frames"
            5. Get instant liveness detection results
            
            **Webcam Tips:**
            - Ensure good lighting
            - Keep face centered
            - Maintain steady position
            - Natural expressions work best
            """)
    
    # ============== TAB 4: BATCH PROCESSING ==============
    with tab4:
        st.header("📊 Batch Processing")
        
        st.markdown("""
        Process multiple images at once and get detailed statistics.
        """)
        
        uploaded_images = st.file_uploader(
            "Upload multiple images",
            type=["jpg", "jpeg", "png", "bmp"],
            accept_multiple_files=True,
            help="Select multiple images for batch processing"
        )
        
        if uploaded_images:
            st.info(f"📁 {len(uploaded_images)} image(s) selected")
            
            if st.button("🔍 Process All Images"):
                results_list = []
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for idx, image_file in enumerate(uploaded_images):
                    status_text.text(f"Processing image {idx + 1}/{len(uploaded_images)}...")
                    
                    image = Image.open(image_file)
                    image_array = np.array(image)
                    
                    if len(image_array.shape) == 3 and image_array.shape[2] == 3:
                        image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
                    
                    # Quick detection
                    faces = face_detector.detect_faces(image_array)
                    
                    if len(faces) > 0:
                        inference = EnhancedLivenessInference(model, device)
                        batch_results = inference.predict_batch_with_features(
                            [image_array], 
                            [[faces[0]['bbox']]]
                        )
                        
                        results_list.append({
                            'filename': image_file.name,
                            'faces': len(faces),
                            'predictions': batch_results['predictions'],
                            'confidences': batch_results['confidences']
                        })
                    
                    progress_bar.progress((idx + 1) / len(uploaded_images))
                
                progress_bar.empty()
                status_text.empty()
                
                # Display results
                st.subheader("📊 Batch Results")
                
                live_count = sum(1 for r in results_list for p in r['predictions'] if p == "Live")
                fake_count = sum(len(r['predictions']) - sum(1 for p in r['predictions'] if p == "Live") 
                               for r in results_list)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Images", len(results_list))
                with col2:
                    st.metric("Live Detections", live_count)
                with col3:
                    st.metric("Fake Detections", fake_count)
                
                # Detailed results table
                st.dataframe(
                    [
                        {
                            'Filename': r['filename'],
                            'Faces': r['faces'],
                            'Predictions': ', '.join(r['predictions']),
                            'Avg Confidence': f"{np.mean(r['confidences']):.2%}"
                        }
                        for r in results_list
                    ],
                    use_container_width=True
                )
    
    # ============== TAB 5: ABOUT ==============
    with tab5:
        st.header("ℹ️ About This Application")
        
        st.markdown("""
        ### 🎯 Purpose
        This application detects facial liveness - distinguishing real faces from spoofed/fake ones.
        This is crucial for anti-spoofing in security systems.
        
        ### 🧠 Models Available
        
        **1. Custom CNN Model**
        - Lightweight and fast
        - Specialized for liveness detection
        - Input size: 300x300
        - Optimized for real-time processing
        
        **2. EfficientNet-B0**
        - Pre-trained on ImageNet
        - More robust generalization
        - Input size: 224x224
        - Better handling of diverse faces
        
        ### 🔍 How It Works
        
        1. **Face Detection**: MediaPipe detects faces in input
        2. **Preprocessing**: Faces are extracted and normalized
        3. **Model Inference**: Selected model predicts liveness
        4. **Aggregation**: Results are combined for final verdict
        
        ### 📈 Features
        
        - ✅ Real-time face detection
        - ✅ Multiple input sources (image, video, webcam)
        - ✅ Batch processing support
        - ✅ Multi-face handling
        - ✅ Confidence scoring
        - ✅ Frame-by-frame video analysis
        
        ### 🎓 Technical Details
        
        - **Framework**: PyTorch
        - **Face Detection**: MediaPipe
        - **UI**: Streamlit
        - **Device Support**: CPU and GPU (CUDA)
        
        ### ⚠️ Limitations
        
        - Performance depends on face quality
        - May struggle with extreme angles
        - Lighting conditions affect accuracy
        - Very high-quality deepfakes might fool the model
        
        ### 📝 Tips for Best Results
        
        1. **Good Lighting**: Avoid shadows on face
        2. **Clear Face**: Ensure face is not obscured
        3. **Still Position**: Keep head steady
        4. **Quality Input**: Use clear, high-resolution images/video
        5. **Multiple Frames**: Videos provide better confidence
        
        ---
        
        **Version**: 1.0.0  
        **Last Updated**: January 2026
        """)
        
        if st.checkbox("Show Technical Details"):
            st.subheader("Model Architecture")
            st.text(f"Selected Model: {model_type}")
            st.text(f"Device: {device}")
            st.text(f"Model Parameters: {sum(p.numel() for p in model.parameters()):,}")

if __name__ == "__main__":
    main()
