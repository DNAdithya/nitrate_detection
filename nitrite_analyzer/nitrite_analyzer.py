import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.spatial.distance import euclidean
import colorsys
import os
from PIL import Image, ImageDraw, ImageFont


class NitriteTestAnalyzer:
    def __init__(self, yolo_model_path='yolov8n.pt'):
        """
        Initialize the Nitrite Test Analyzer
        
        Args:
            yolo_model_path: Path to YOLO model (will download if not exists)
        """
        self.model = YOLO(yolo_model_path)
        
        # Reference color chart values (RGB values for each concentration)
        # These are approximate values - you may need to calibrate based on your specific test kit
        self.reference_colors = {
            0.0: [240, 240, 240],   # Off-white
            0.5: [255, 220, 255],   # Very light pink
            1.0: [255, 182, 193],   # Light pink
            2.0: [255, 105, 180],   # Medium pink
            3.0: [255, 20, 147],    # Deep pink
            5.0: [190, 40, 120]     # Dark magenta
        }

        
        # Define class names for detected objects
        self.class_names = {
            0: 'test_tube'

        }
        
    def detect_objects(self, image):
        """
        Detect objects in the image using YOLO
        
        Args:
            image: Input image (PIL Image or numpy array)
            
        Returns:
            boxes: List of bounding boxes
            scores: List of confidence scores
            detected_classes: List of detected class indices
        """
        # Convert PIL Image to numpy array if needed
        if isinstance(image, Image.Image):
            image_array = np.array(image)
        else:
            image_array = image
            
        # Run YOLO detection
        results = self.model(image_array, conf=0.5)
        
        boxes = []
        scores = []
        detected_classes = []
        
        # Extract detection results
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    boxes.append([int(x1), int(y1), int(x2), int(y2)])
                    
                    # Get confidence score
                    scores.append(float(box.conf[0].cpu().numpy()))
                    
                    # Get class index
                    detected_classes.append(int(box.cls[0].cpu().numpy()))
        
        return boxes, scores, detected_classes
    
    def draw_detected_boxes(self, image, boxes, scores, detected_classes, class_names=None):
        """
        Draw detected bounding boxes on the image
        
        Args:
            image: Input image (PIL Image)
            boxes: List of bounding boxes
            scores: List of confidence scores
            detected_classes: List of detected class indices
            class_names: Dictionary mapping class indices to names
            
        Returns:
            annotated_image: PIL Image with drawn boxes
        """
        if class_names is None:
            class_names = self.class_names
            
        # Convert to PIL Image if needed
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # Create a copy of the image
        annotated_image = image.copy()
        draw = ImageDraw.Draw(annotated_image)
        
        # Define colors for different classes
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
        
        
        for i, (box, score, class_idx) in enumerate(zip(boxes, scores, detected_classes)):
            x1, y1, x2, y2 = box
            color = colors[class_idx % len(colors)]
            
            # Draw bounding box
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            
            # Draw label
            class_name = class_names.get(class_idx, f'Class_{class_idx}')
            label = f'{class_name}: {score:.2f}'
            print(f"Detected {class_name} with confidence {score:.2f} at [{x1}, {y1}, {x2}, {y2}]")
            # Try to use a font, fall back to default if not available
            try:
                font = ImageFont.truetype("arial.ttf", 16)
            except:
                font = ImageFont.load_default()
            
            # Get text size using textbbox (available in Pillow >= 8.0)
            bbox = draw.textbbox((0, 0), label, font=font)
            print(f"bbox: {bbox}")
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]

            # Draw text background rectangle
            draw.rectangle(
                [x1, y1 - text_height - 5, x1 + text_width + 5, y1],
                fill=color, outline=color
            )

            # Draw text on top of the background
            draw.text((x1 + 2, y1 - text_height - 3), label, fill='white', font=font)

        
        return annotated_image
    
    def extract_liquid_region(self, test_tube_region, liquid_ratio=0.30):
        """
        Extract only the liquid region from test tube (bottom portion)
        
        Args:
            test_tube_region: Test tube image region
            liquid_ratio: Ratio of bottom region to consider (0.6 = bottom 60%)
            
        Returns:
            liquid_region: Cropped liquid region
        """
        if isinstance(test_tube_region, Image.Image):
            test_tube_region = np.array(test_tube_region)
        
        height, width = test_tube_region.shape[:2]
        
        # Calculate the starting point for liquid region (from bottom)
        liquid_start = int(height * (1 - liquid_ratio))
        
        # Extract bottom portion where liquid is present
        liquid_region = test_tube_region[liquid_start:, :]
        
        return liquid_region

    def extract_dominant_color_advanced(self, image_region, method='kmeans', k=5):
        """
        Advanced dominant color extraction with multiple methods
        
        Args:
            image_region: Image region to analyze
            method: 'kmeans', 'histogram', 'median', or 'combined'
            k: Number of clusters for K-means
            
        Returns:
            dominant_color: RGB values of dominant color
            confidence: Confidence score of the extraction
        """
        # Convert PIL Image to numpy array if needed
        if isinstance(image_region, Image.Image):
            image_region = np.array(image_region)
        
        # Get liquid region (bottom 60%)
        liquid_region = self.extract_liquid_region(image_region, liquid_ratio=0.30)
        
        if liquid_region.size == 0:
            return [0, 0, 0], 0.0
        
        # Ensure RGB format
        if len(liquid_region.shape) == 3 and liquid_region.shape[2] == 3:
            pixels = liquid_region.reshape(-1, 3)
        else:
            return [0, 0, 0], 0.0
        
        # Advanced preprocessing
        pixels = self._preprocess_pixels(pixels)
        
        if len(pixels) == 0:
            return [0, 0, 0], 0.0
        
        if method == 'kmeans':
            return self._extract_color_kmeans(pixels, k)
        elif method == 'histogram':
            return self._extract_color_histogram(pixels)
        elif method == 'median':
            return self._extract_color_median(pixels)
        elif method == 'combined':
            return self._extract_color_combined(pixels, k)
        else:
            return self._extract_color_kmeans(pixels, k)

    def _preprocess_pixels(self, pixels, trim_ratio=0.10):
        """
        Preprocess pixels to remove border noise and improve color extraction.
        
        Args:
            pixels (ndarray): Input image pixels (H x W x 3)
            trim_ratio (float): Fraction of width to trim from each side (default 5%)

        Returns:
            final_pixels (ndarray): Filtered pixels array (N x 3)
        """
        if pixels.ndim == 3:
            h, w, _ = pixels.shape
            trim_px = int(w * trim_ratio)

            # Trim left and right borders
            pixels = pixels[:, trim_px: w - trim_px, :]

            # Flatten to list of RGB pixels
            pixels = pixels.reshape(-1, 3)

        if len(pixels) == 0:
            return pixels

        # Remove very dark and very bright pixels
        pixel_sum = np.sum(pixels, axis=1)
        brightness = np.mean(pixels, axis=1)
        valid_mask = (
            (pixel_sum > 30) &
            (pixel_sum < 720) &
            (brightness > 40) &
            (brightness < 220)
        )
        filtered_pixels = pixels[valid_mask]

        if len(filtered_pixels) == 0:
            return filtered_pixels

        # Convert to HSV and remove low-saturation pixels
        rgb_norm = filtered_pixels / 255.0
        r, g, b = rgb_norm[:, 0], rgb_norm[:, 1], rgb_norm[:, 2]
        maxc = np.max(rgb_norm, axis=1)
        minc = np.min(rgb_norm, axis=1)
        delta = maxc - minc
        saturation = np.zeros_like(maxc)
        nonzero_mask = maxc != 0
        saturation[nonzero_mask] = delta[nonzero_mask] / maxc[nonzero_mask]
        final_pixels = filtered_pixels[saturation > 0.2]

        return filtered_pixels #final_pixels


    def _extract_color_kmeans(self, pixels, k):
        """
        Extract dominant color using K-means clustering
        """
        if len(pixels) < k:
            return np.mean(pixels, axis=0).astype(int), 0.5
        
        # Apply K-means clustering
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(pixels)
        
        # Get cluster centers and their sizes
        colors = kmeans.cluster_centers_
        labels = kmeans.labels_
        
        # Calculate cluster sizes and confidence
        unique_labels, counts = np.unique(labels, return_counts=True)
        
        # Find the most common cluster
        dominant_idx = np.argmax(counts)
        dominant_color = colors[unique_labels[dominant_idx]]
        
        # Calculate confidence based on cluster dominance
        confidence = counts[dominant_idx] / len(pixels)
        
        return dominant_color.astype(int), confidence

    def _extract_color_histogram(self, pixels):
        """
        Extract dominant color using color histogram
        """
        # Quantize colors to reduce noise
        quantized = (pixels // 32) * 32
        
        # Count color occurrences
        colors, counts = np.unique(quantized, axis=0, return_counts=True)
        
        # Find most common color
        dominant_idx = np.argmax(counts)
        dominant_color = colors[dominant_idx]
        
        # Calculate confidence
        confidence = counts[dominant_idx] / len(pixels)
        
        return dominant_color.astype(int), confidence

    def _extract_color_median(self, pixels):
        """
        Extract dominant color using median filtering
        """
        # Use median for each channel to reduce noise
        dominant_color = np.median(pixels, axis=0)
        
        # Calculate confidence based on consistency
        distances = np.linalg.norm(pixels - dominant_color, axis=1)
        similar_pixels = np.sum(distances < 50)  # Pixels within 50 units
        confidence = similar_pixels / len(pixels)
        
        return dominant_color.astype(int), confidence

    def _extract_color_combined(self, pixels, k):
        """
        Extract dominant color using combined approach
        """
        # Get results from multiple methods
        kmeans_color, kmeans_conf = self._extract_color_kmeans(pixels, k)
        hist_color, hist_conf = self._extract_color_histogram(pixels)
        median_color, median_conf = self._extract_color_median(pixels)
        
        # Weight the results based on confidence
        total_weight = kmeans_conf + hist_conf + median_conf
        
        if total_weight == 0:
            return [0, 0, 0], 0.0
        
        # Weighted average of colors
        combined_color = (
            kmeans_color * kmeans_conf +
            hist_color * hist_conf +
            median_color * median_conf
        ) / total_weight
        
        # Average confidence
        avg_confidence = total_weight / 3
        
        return combined_color.astype(int), avg_confidence

    def extract_liquid_color_with_validation(self, test_tube_region, methods=['kmeans'], k=3):
        """
        Extract liquid color with validation across multiple methods
        
        Args:
            test_tube_region: Test tube image region
            methods: List of methods to use for validation
            k: Number of clusters for K-means
            
        Returns:
            final_color: Most reliable color
            confidence: Overall confidence score
            method_results: Results from all methods
        """
        method_results = {}
        
        for method in methods:
            color, conf = self.extract_dominant_color_advanced(test_tube_region, method=method, k=k)
            method_results[method] = {'color': color, 'confidence': conf}
        
        # Find the method with highest confidence
        best_method = max(method_results.keys(), key=lambda m: method_results[m]['confidence'])
        best_result = method_results[best_method]
        
        # Validate consistency across methods
        colors = [result['color'] for result in method_results.values()]
        if len(colors) > 1:
            # Check if colors are similar (within 30 units in RGB space)
            avg_color = np.mean(colors, axis=0)
            distances = [np.linalg.norm(color - avg_color) for color in colors]
            consistency = 1.0 - (np.mean(distances) / 100)  # Normalize to 0-1
            
            # Adjust confidence based on consistency
            final_confidence = best_result['confidence'] * max(0.5, consistency)
        else:
            final_confidence = best_result['confidence']
        print("best_result color", best_result['color'])
        return best_result['color'], final_confidence, method_results
    
    def rgb_to_lab(self, rgb):
        """
        Convert RGB to LAB color space for better color comparison
        
        Args:
            rgb: RGB color values
            
        Returns:
            lab: LAB color values
        """
        rgb_normalized = np.array(rgb) / 255.0
        
        # Convert to XYZ
        def rgb_to_xyz(c):
            if c > 0.04045:
                return ((c + 0.055) / 1.055) ** 2.4
            else:
                return c / 12.92
        
        r, g, b = [rgb_to_xyz(c) for c in rgb_normalized]
        
        # Observer = 2°, Illuminant = D65
        x = r * 0.4124 + g * 0.3576 + b * 0.1805
        y = r * 0.2126 + g * 0.7152 + b * 0.0722
        z = r * 0.0193 + g * 0.1192 + b * 0.9505
        
        # Normalize for D65 illuminant
        x = x / 0.95047
        y = y / 1.00000
        z = z / 1.08883
        
        # Convert to LAB
        def xyz_to_lab(c):
            if c > 0.008856:
                return c ** (1/3)
            else:
                return (7.787 * c) + (16/116)
        
        fx = xyz_to_lab(x)
        fy = xyz_to_lab(y)
        fz = xyz_to_lab(z)
        
        l = (116 * fy) - 16
        a = 500 * (fx - fy)
        b = 200 * (fy - fz)
        
        return [l, a, b]
    
    def color_similarity(self, color1, color2, method='lab'):
        """
        Calculate color similarity between two colors
        
        Args:
            color1, color2: RGB color values
            method: Similarity method ('rgb', 'hsv', 'lab')
            
        Returns:
            similarity: Similarity score (lower is more similar)
        """
        if method == 'rgb':
            return euclidean(color1, color2)
        
        elif method == 'hsv':
            hsv1 = colorsys.rgb_to_hsv(color1[0]/255, color1[1]/255, color1[2]/255)
            hsv2 = colorsys.rgb_to_hsv(color2[0]/255, color2[1]/255, color2[2]/255)
            return euclidean(hsv1, hsv2)
        
        elif method == 'lab':
            lab1 = self.rgb_to_lab(color1)
            lab2 = self.rgb_to_lab(color2)
            return euclidean(lab1, lab2)
    
    def match_color_to_reference(self, test_color):
        """
        Match test color to reference chart
        
        Args:
            test_color: RGB values of test color
            
        Returns:
            best_match: Concentration level with best match
            confidence: Confidence score (0-1)
            all_similarities: Dictionary of all similarities
        """
        similarities = {}
        
        for concentration, ref_color in self.reference_colors.items():
            similarity = self.color_similarity(test_color, ref_color, method='lab')
            similarities[concentration] = similarity
        
        # Find best match
        best_match = min(similarities.keys(), key=lambda x: similarities[x])
        best_similarity = similarities[best_match]
        
        # Calculate confidence (inverse of similarity, normalized)
        max_similarity = max(similarities.values())
        confidence = 1.0 - (best_similarity / max_similarity) if max_similarity > 0 else 1.0
        confidence = max(0, min(1, confidence))
        
        return best_match, confidence, similarities
    
    def detect_test_tube_region(self, image, boxes, scores, detected_classes, confidence_threshold=0.5, test_tube_class_id=0):
        """
        Detect test tube region using YOLO detection results
        
        Args:
            image: Input image (PIL Image or numpy array)
            boxes: YOLO bounding boxes (numpy array or list)
            scores: YOLO confidence scores (numpy array or list)
            detected_classes: YOLO detected class IDs (numpy array or list)
            confidence_threshold: Minimum confidence threshold for detection
            test_tube_class_id: Class ID for test tube in your YOLO model
            
        Returns:
            test_tube_region: Cropped test tube region
            bbox: Bounding box coordinates [x, y, w, h]
        """
        # Convert PIL Image to numpy array if needed
        if isinstance(image, Image.Image):
            image_array = np.array(image)
        else:
            image_array = image
        
        # Convert to numpy arrays if they aren't already
        if not isinstance(boxes, np.ndarray):
            boxes = np.array(boxes)
        if not isinstance(scores, np.ndarray):
            scores = np.array(scores)
        if not isinstance(detected_classes, np.ndarray):
            detected_classes = np.array(detected_classes)
        
        # Filter detections based on confidence threshold and test tube class
        valid_detections = (scores >= confidence_threshold) & (detected_classes == test_tube_class_id)
        
        if not np.any(valid_detections):
            print(f"No test tube detected with confidence >= {confidence_threshold}")
            return None, None
        
        # Get valid boxes and scores
        valid_boxes = boxes[valid_detections]
        valid_scores = scores[valid_detections]
        
        # Find the detection with highest confidence
        best_detection_idx = np.argmax(valid_scores)
        best_box = valid_boxes[best_detection_idx]
        best_score = valid_scores[best_detection_idx]
        
        print(f"Best test tube detection confidence: {best_score:.3f}")
        
        # Handle different box formats
        if len(best_box) == 4:
            # Handle both [x, y, w, h] and [x1, y1, x2, y2] formats
            if best_box[2] > image_array.shape[1] or best_box[3] > image_array.shape[0]:
                # Likely [x1, y1, x2, y2] format
                x1, y1, x2, y2 = best_box
                x, y, w, h = int(x1), int(y1), int(x2 - x1), int(y2 - y1)
            else:
                # Likely [x, y, w, h] format
                x, y, w, h = map(int, best_box)
        else:
            print(f"Unexpected box format: {best_box}")
            return None, None
        
        # Ensure coordinates are within image bounds
        height, width = image_array.shape[:2]
        x = max(0, min(x, width - 1))
        y = max(0, min(y, height - 1))
        w = max(1, min(w, width - x))
        h = max(1, min(h, height - y))
        
        # Crop the test tube region
        test_tube_region = image_array[y:y+h, x:x+w]
        bbox = [x, y, w, h]
        
        return test_tube_region, bbox
    
    def analyze_image(self, image_path,boxes, scores, detected_classes):
        """
        Main analysis function
        
        Args:
            image_path: Path to input image or PIL Image object
            
        Returns:
            results: Dictionary containing analysis results
            image: Processed image
        """
        # Load image
        if isinstance(image_path, str):
            if not os.path.exists(image_path):
                raise ValueError(f"Image file not found: {image_path}")
            image = Image.open(image_path)
        elif isinstance(image_path, Image.Image):
            image = image_path
        else:
            raise ValueError("Input must be a file path or PIL Image object")
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        image_array = np.array(image)
        
        # Detect test tube region
        test_tube_region, tube_bbox = self.detect_test_tube_region(image_array,boxes, scores, detected_classes)

        if test_tube_region is None:
            # If no specific region detected, use the whole image
            test_tube_region = image_array
            tube_bbox = [0, 0, image_array.shape[1], image_array.shape[0]]
        
        # Extract dominant color from test tube
        test_color, final_confidence, method_results= self.extract_liquid_color_with_validation(test_tube_region)
        
        # Match color to reference chart
        best_match, confidence, similarities = self.match_color_to_reference(test_color)
        
        results = {
            'nitrite_level': best_match,
            'confidence': confidence,
            'test_color_rgb': test_color.tolist(),
            'similarities': similarities,
            'tube_bbox': tube_bbox,
            'image_shape': image_array.shape
        }
        
        return results, image
    
    def visualize_results(self, image, results):
        """
        Visualize analysis results
        
        Args:
            image: Input image (PIL Image)
            results: Analysis results
        """
        # Convert PIL Image to numpy array for matplotlib
        if isinstance(image, Image.Image):
            image_array = np.array(image)
        else:
            image_array = image
            
        fig, axes = plt.subplots(2, 2, figsize=(8, 8))
        
        # Original image with bounding boxes
        ax1 = axes[0, 0]
        ax1.imshow(image_array)
        ax1.set_title('Original Image with Detections')
        
        # Draw bounding boxes
        if results['tube_bbox']:
            x, y, w, h = results['tube_bbox']
            rect = plt.Rectangle((x, y), w, h, fill=False, color='red', linewidth=2)
            ax1.add_patch(rect)
            ax1.text(x, y-10, 'Test Region', color='red', fontsize=12, fontweight='bold')
        
        ax1.axis('off')
        
        # Test color vs reference colors
        ax2 = axes[0, 1]
        colors = []
        labels = []
        
        # Add test color
        test_color_normalized = np.array(results['test_color_rgb']) / 255.0
        colors.append(test_color_normalized)
        labels.append(f"Test Color\n(Detected)")
        
        # Add reference colors
        for concentration, ref_color in self.reference_colors.items():
            ref_color_normalized = np.array(ref_color) / 255.0
            colors.append(ref_color_normalized)
            labels.append(f"{concentration}\nmg/L")
        
        # Create color swatches
        for i, (color, label) in enumerate(zip(colors, labels)):
            rect = plt.Rectangle((i, 0), 1, 1, facecolor=color, edgecolor='black')
            ax2.add_patch(rect)
            ax2.text(i + 0.5, -0.3, label, ha='center', va='top', fontsize=10)
        
        ax2.set_xlim(0, len(colors))
        ax2.set_ylim(-0.5, 1.5)
        ax2.set_title('Color Comparison')
        ax2.axis('off')
        
        # Similarity scores
        ax3 = axes[1, 0]
        concentrations = list(results['similarities'].keys())
        similarities = list(results['similarities'].values())
        
        bars = ax3.bar(range(len(concentrations)), similarities)
        ax3.set_xlabel('Concentration (mg/L)')
        ax3.set_ylabel('Color Difference (Lower = Better Match)')
        ax3.set_title('Color Similarity Scores')
        ax3.set_xticks(range(len(concentrations)))
        ax3.set_xticklabels([str(c) for c in concentrations])
        
        # Highlight best match
        best_idx = concentrations.index(results['nitrite_level'])
        bars[best_idx].set_color('red')
        
        # Results summary
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        result_text = f"""
        NITRITE TEST RESULTS
        
        Detected Level: {results['nitrite_level']} mg/L
        Confidence: {results['confidence']:.2f}
        
        Test Color (RGB): {results['test_color_rgb']}
        
        Color Analysis Method: LAB color space
        
        Interpretation:
        """
        
        if results['nitrite_level'] == 0.0:
            result_text += "• Safe - No nitrite detected"
        elif results['nitrite_level'] <= 1.0:
            result_text += "• Low level - Monitor regularly"
        elif results['nitrite_level'] <= 2.0:
            result_text += "• Moderate level - Take action"
        else:
            result_text += "• High level - Immediate attention needed"
        
        ax4.text(0.1, 0.9, result_text, transform=ax4.transAxes, fontsize=12,
                verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
        
        plt.tight_layout()
        plt.show()

# Example usage
def main():
    # Initialize analyzer
    analyzer = NitriteTestAnalyzer()
    
    # Analyze image
    image_path = "nitrite_test_image.jpg"  # Replace with your image path
    
    try:
        results, image = analyzer.analyze_image(image_path)
        
        # Print results
        print("=== NITRITE TEST ANALYSIS RESULTS ===")
        print(f"Detected Nitrite Level: {results['nitrite_level']} mg/L")
        print(f"Confidence: {results['confidence']:.2f}")
        print(f"Test Color (RGB): {results['test_color_rgb']}")
        print("\nColor Similarity Scores:")
        for concentration, similarity in results['similarities'].items():
            print(f"  {concentration} mg/L: {similarity:.2f}")
        
        # Visualize results
        analyzer.visualize_results(image, results)
        
    except Exception as e:
        print(f"Error analyzing image: {e}")
        print("Make sure you have installed required packages:")
        print("pip install ultralytics opencv-python matplotlib scikit-learn scipy pillow")

if __name__ == "__main__":
    main()