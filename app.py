import streamlit as st
import sqlite3
import os
from datetime import datetime
import base64
from PIL import Image
import io
import numpy as np
import matplotlib.pyplot as plt
from nitrite_analyzer.nitrite_analyzer import NitriteTestAnalyzer

# Page configuration
st.set_page_config(
    page_title="Nitrite Color Identification System",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #2E86AB;
        padding: 1rem 0;
        border-bottom: 3px solid #2E86AB;
        margin-bottom: 2rem;
    }
    
    .section-header {
        color: #A23B72;
        border-left: 4px solid #A23B72;
        padding-left: 1rem;
        margin: 1.5rem 0 1rem 0;
    }
    
    .info-box {
        background-color: #f0f8ff;
        border-left: 4px solid #2E86AB;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0 8px 8px 0;
    }
    
    .success-box {
        background-color: #f0fff0;
        border: 1px solid #90EE90;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .error-box {
        background-color: #fff0f0;
        border: 1px solid #ffcccb;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize the analyzer (cached to avoid reloading)
@st.cache_resource
def load_analyzer():
    """Load the nitrite analyzer with caching"""
    try:
        # Check if custom weights exist, otherwise use default
        weights_path = "weights/best.pt"
        if not os.path.exists(weights_path):
            weights_path = 'yolov8n.pt'  # Default YOLO weights
        
        return NitriteTestAnalyzer(yolo_model_path=weights_path)
    except Exception as e:
        st.error(f"Error loading analyzer: {str(e)}")
        return None

# Initialize database
def init_database():
    """Initialize the SQLite database with proper table structure"""
    try:
        conn = sqlite3.connect('nitrite.db')
        cursor = conn.cursor()
        
        # First, check if table exists and get its structure
        cursor.execute("PRAGMA table_info(nitrite)")
        columns = [column[1] for column in cursor.fetchall()]
        
        if not columns:  # Table doesn't exist
            # Create table with all required columns
            cursor.execute('''
                CREATE TABLE nitrite (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    image_name TEXT NOT NULL,
                    unit TEXT NOT NULL,
                    concentration REAL,
                    confidence REAL,
                    test_color_rgb TEXT,
                    upload_date TEXT NOT NULL,
                    image_data BLOB
                )
            ''')
            print("Created new nitrite table with all columns")
        else:
            # Table exists, check for missing columns and add them
            required_columns = {
                'confidence': 'REAL',
                'test_color_rgb': 'TEXT',
                'image_data': 'BLOB'
            }
            
            for col_name, col_type in required_columns.items():
                if col_name not in columns:
                    try:
                        cursor.execute(f'ALTER TABLE nitrite ADD COLUMN {col_name} {col_type}')
                        print(f"Added missing column: {col_name}")
                    except sqlite3.OperationalError as e:
                        print(f"Error adding column {col_name}: {e}")
        
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        st.error(f"Database initialization error: {str(e)}")
        return False
    
# Function to save results to database
def save_to_database(image_name, unit, results, image_data):
    """Save analysis results and image data to database"""
    try:
        conn = sqlite3.connect('nitrite.db')
        cursor = conn.cursor()
        
        # Convert image to bytes
        img_bytes = image_data.getvalue()
        
        # Insert data with all required fields
        cursor.execute('''
            INSERT INTO nitrite (image_name, unit, concentration, confidence, test_color_rgb, upload_date, image_data)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            image_name,
            unit,
            results['nitrite_level'],
            results['confidence'],
            str(results['test_color_rgb']),
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            img_bytes
        ))
        
        conn.commit()
        conn.close()
        st.success("Results saved to database successfully!")
        return True
    except Exception as e:
        st.error(f"Database save error: {str(e)}")
        return False

# Function to load history from database
def load_history():
    """Load history from database"""
    try:
        conn = sqlite3.connect('nitrite.db')
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM nitrite ORDER BY id DESC")
        data = cursor.fetchall()
        
        conn.close()
        return data
    except Exception as e:
        st.error(f"Database load error: {str(e)}")
        return []

# Function to convert PIL image to bytes
def pil_to_bytes(pil_image):
    """Convert PIL image to bytes"""
    img_byte_arr = io.BytesIO()
    pil_image.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    return img_byte_arr

# Main application
def main():
    # Initialize database
    if not init_database():
        st.stop()
    
    # Load analyzer
    analyzer = load_analyzer()
    if analyzer is None:
        st.error("Failed to load the nitrite analyzer. Please check your model weights.")
        st.stop()
    
    # Main title
    st.markdown('<h1 class="main-header">🧪 Nitrite Color Identification System</h1>', unsafe_allow_html=True)
    
    # Sidebar for navigation
    with st.sidebar:
        st.header("Navigation")
        page = st.radio("Select Page:", ["Upload & Test", "History", "About"])
        
        st.markdown("---")
        st.markdown("### 🔬 Test Parameters")
        st.markdown("**Reference Levels:**")
        st.markdown("- 0.0 mg/L: Safe")
        st.markdown("- 0.5-1.0 mg/L: Low")
        st.markdown("- 1.0-2.0 mg/L: Moderate")
        st.markdown("- 2.0+ mg/L: High")
    
    # Main content based on page selection
    if page == "Upload & Test":
        upload_page(analyzer)
    elif page == "History":
        history_page()
    else:
        about_page()

def upload_page(analyzer):
    """Main upload and testing page"""
    st.markdown('<h2>📤 Upload Image for Testing</h2>', unsafe_allow_html=True)
    
    # Create two columns for better layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### Image Upload")
        uploaded_image = st.file_uploader(
            "Choose an image file",
            type=["jpg", "jpeg", "png", "bmp", "tiff"],
            help="Supported formats: JPG, JPEG, PNG, BMP, TIFF"
        )
        
        if uploaded_image is not None:
            # Display uploaded image
            try:
                image = Image.open(uploaded_image)
                st.image(image, caption=f'Uploaded Image: {uploaded_image.name}', use_column_width=True)
                
                # Image information
                st.markdown("**Image Details:**")
                st.write(f"- **Name:** {uploaded_image.name}")
                st.write(f"- **Size:** {uploaded_image.size} bytes")
                st.write(f"- **Dimensions:** {image.size[0]} x {image.size[1]} pixels")
                st.write(f"- **Format:** {image.format}")
                
            except Exception as e:
                st.error(f"Error loading image: {str(e)}")
    
    with col2:
        st.markdown("### Test Parameters")
        
        # Unit selection
        unit = st.radio(
            'Select measurement unit:',
            ('mg/L', 'ppm'),
            help="Choose the unit for nitrite concentration measurement"
        )
        
        # Analysis options
        st.markdown("### Analysis Options")
        show_detections = st.checkbox("Show object detections", value=True)
        show_color_analysis = st.checkbox("Show color analysis", value=True)
        
        # Submit button
        st.markdown("---")
        submit_button = st.button("🔍 Analyze Image", type="primary", use_container_width=True)
    
    # Process submission
    if submit_button:
        if uploaded_image is not None:
            try:
                with st.spinner("Analyzing image..."):
                    # Load image
                    image = Image.open(uploaded_image)
                    
                    # Analyze image using the nitrite analyzer
                    results, processed_image = analyzer.analyze_image(image)
                    
                    # Get object detections
                    boxes, scores, detected_classes = analyzer.detect_objects(image)
                    
                    # Draw detected boxes on the image
                    if show_detections and boxes:
                        analyzed_image = analyzer.draw_detected_boxes(image, boxes, scores, detected_classes)
                        st.markdown("### 🎯 Object Detection Results")
                        st.image(analyzed_image, caption="Image with Detections", use_column_width=True)
                    
                    # Display analysis results
                    st.markdown("### 📊 Analysis Results")
                    
                    # Create metrics
                    results_col1, results_col2, results_col3 = st.columns(3)
                    
                    with results_col1:
                        st.metric(
                            "Nitrite Level", 
                            f"{results['nitrite_level']} {unit}",
                            help="Detected nitrite concentration"
                        )
                    
                    with results_col2:
                        st.metric(
                            "Confidence", 
                            f"{results['confidence']:.1%}",
                            help="Analysis confidence level"
                        )
                    
                    with results_col3:
                        # Interpretation
                        level = results['nitrite_level']
                        if level == 0.0:
                            interpretation = "Safe ✅"
                            delta_color = "normal"
                        elif level <= 1.0:
                            interpretation = "Low ⚠️"
                            delta_color = "normal"
                        elif level <= 2.0:
                            interpretation = "Moderate ⚠️"
                            delta_color = "inverse"
                        else:
                            interpretation = "High ❌"
                            delta_color = "inverse"
                        
                        st.metric(
                            "Risk Level", 
                            interpretation,
                            help="Health risk assessment"
                        )
                    
                    # Color analysis
                    if show_color_analysis:
                        st.markdown("### 🎨 Color Analysis")
                        
                        color_col1, color_col2 = st.columns(2)
                        
                        with color_col1:
                            st.markdown("**Detected Color (RGB):**")
                            rgb_values = results['test_color_rgb']
                            st.write(f"Red: {rgb_values[0]}, Green: {rgb_values[1]}, Blue: {rgb_values[2]}")
                            
                            # Show color swatch
                            color_normalized = np.array(rgb_values) / 255.0
                            fig, ax = plt.subplots(1, 1, figsize=(3, 1))
                            ax.imshow([[color_normalized]])
                            ax.set_title("Detected Color")
                            ax.axis('off')
                            st.pyplot(fig)
                            plt.close()
                        
                        with color_col2:
                            st.markdown("**Similarity Scores:**")
                            similarities = results['similarities']
                            
                            # Create a bar chart of similarities
                            concentrations = list(similarities.keys())
                            similarity_values = list(similarities.values())
                            
                            fig, ax = plt.subplots(figsize=(6, 4))
                            bars = ax.bar(range(len(concentrations)), similarity_values)
                            ax.set_xlabel('Concentration (mg/L)')
                            ax.set_ylabel('Color Difference')
                            ax.set_title('Color Similarity Analysis')
                            ax.set_xticks(range(len(concentrations)))
                            ax.set_xticklabels([str(c) for c in concentrations])
                            
                            # Highlight best match
                            best_idx = concentrations.index(results['nitrite_level'])
                            bars[best_idx].set_color('red')
                            
                            st.pyplot(fig)
                            plt.close()
                    
                    # Save results to database
                    image_bytes = pil_to_bytes(image)
                    if save_to_database(uploaded_image.name, unit, results, image_bytes):
                        st.success("✅ Analysis complete! Results saved to database.")
                    else:
                        st.warning("⚠️ Analysis complete, but failed to save to database.")
                    
                    # Health recommendations
                    st.markdown("### 💡 Recommendations")
                    level = results['nitrite_level']
                    
                    if level == 0.0:
                        st.success("✅ **Safe Level**: No action needed. Continue regular monitoring.")
                    elif level <= 1.0:
                        st.warning("⚠️ **Low Level**: Monitor more frequently. Consider water source investigation.")
                    elif level <= 2.0:
                        st.warning("⚠️ **Moderate Level**: Take action to reduce nitrite levels. Consider water treatment.")
                    else:
                        st.error("❌ **High Level**: Immediate attention required! Do not consume water. Contact water treatment professional.")
                    
            except Exception as e:
                st.error(f"Error analyzing image: {str(e)}")
                st.error("Please make sure the image is clear and contains a nitrite test strip or sample.")
        else:
            st.markdown("""
            <div class="error-box">
                <h4>❌ Error</h4>
                <p>Please upload an image before submitting.</p>
            </div>
            """, unsafe_allow_html=True)

def history_page():
    """History page showing previous tests"""
    st.markdown('<h2 class="section-header">📋 Test History</h2>', unsafe_allow_html=True)
    
    # Load history
    history_data = load_history()
    
    if history_data:
        st.success(f"Found {len(history_data)} previous tests")
        
        # Display history in a more organized way
        for i, row in enumerate(history_data):
            # Unpack row data (accounting for new schema)
            test_id, image_name, unit, concentration, confidence, test_color_rgb, upload_date, image_data = row
            
            with st.expander(f"Test #{test_id} - {image_name} ({upload_date})"):
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.write("**Test Details:**")
                    st.write(f"- **ID:** {test_id}")
                    st.write(f"- **Image:** {image_name}")
                    st.write(f"- **Unit:** {unit}")
                    st.write(f"- **Concentration:** {concentration}")
                    # Safe confidence formatting
                    if confidence is not None:
                        try:
                            conf_value = float(confidence)
                            st.write(f"- **Confidence:** {conf_value:.1%}")
                        except (ValueError, TypeError):
                            st.write(f"- **Confidence:** {confidence}")
                    else:
                        st.write("- **Confidence:** N/A")
                    st.write(f"- **Date:** {upload_date}")
                    
                    # Download button for image
                    if image_data:
                        st.download_button(
                            label="Download Image",
                            data=image_data,
                            file_name=image_name,
                            mime="image/png"
                        )
                
                with col2:
                    # Display saved image if available
                    if image_data:
                        try:
                            image = Image.open(io.BytesIO(image_data))
                            st.image(image, caption=f"Test Image: {image_name}", width=300)
                        except Exception as e:
                            st.error(f"Could not display image: {str(e)}")
        
        # Clear history button
        st.markdown("---")
        if st.button("🗑️ Clear All History", type="secondary"):
            if st.checkbox("I confirm I want to delete all history"):
                try:
                    conn = sqlite3.connect('nitrite.db')
                    cursor = conn.cursor()
                    cursor.execute("DELETE FROM nitrite")
                    conn.commit()
                    conn.close()
                    st.success("History cleared successfully!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error clearing history: {str(e)}")
    else:
        st.info("No previous tests found. Upload an image to get started!")

def about_page():
    """About page with system information"""
    st.markdown('<h2>ℹ️ About This System</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    ### 🧪 Nitrite Color Identification System
    
    This application uses advanced computer vision and machine learning to analyze nitrite concentration 
    levels through automated color detection and analysis of test strips or water samples.
    
    #### 🎯 Key Features:
    - **AI-Powered Analysis**: Uses YOLO object detection and color analysis
    - **Multiple Format Support**: JPG, JPEG, PNG, BMP, TIFF
    - **Real-time Processing**: Instant analysis and results
    - **Unit Conversion**: Support for mg/L and ppm measurements
    - **History Tracking**: Save and review previous tests
    - **Confidence Scoring**: Reliability assessment for each analysis
    
    #### 📊 How It Works:
    1. **Object Detection**: Identifies test strips and sample areas using YOLO
    2. **Color Extraction**: Analyzes dominant colors using K-means clustering
    3. **Color Matching**: Compares detected colors to reference standards
    4. **Concentration Estimation**: Provides nitrite level based on color analysis
    5. **Result Validation**: Calculates confidence scores for reliability
    
    #### 🔬 Technical Details:
    - **Framework**: Streamlit for web interface
    - **AI Model**: YOLO v8 for object detection
    - **Color Analysis**: LAB color space for accurate comparison
    - **Database**: SQLite for local storage
    - **Image Processing**: OpenCV, PIL, and scikit-learn
    
    #### 📏 Measurement Standards:
    - **0.0 mg/L**: Safe - No nitrite detected
    - **0.5-1.0 mg/L**: Low level - Monitor regularly
    - **1.0-2.0 mg/L**: Moderate level - Take action
    - **2.0+ mg/L**: High level - Immediate attention needed
    
    #### 📝 Usage Notes:
    - Ensure good lighting when capturing test strip images
    - Hold the camera steady for clear, focused images
    - Position test strips clearly in the frame
    - Results are estimates and should be verified with laboratory tests for critical applications
    - Regular calibration with known standards is recommended
    """)
    
    # System status
    st.markdown("### 🔍 System Status")
    
    status_col1, status_col2, status_col3 = st.columns(3)
    
    with status_col1:
        # Check database connection
        try:
            conn = sqlite3.connect('nitrite.db')
            conn.close()
            db_status = "✅ Connected"
        except:
            db_status = "❌ Error"
        st.metric("Database Status", db_status)
    
    with status_col2:
        history_count = len(load_history())
        st.metric("Total Tests", history_count)
    
    with status_col3:
        # Check if analyzer is loaded
        analyzer = load_analyzer()
        model_status = "✅ Loaded" if analyzer else "❌ Error"
        st.metric("AI Model Status", model_status)

if __name__ == "__main__":
    main()