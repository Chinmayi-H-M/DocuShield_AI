import os
import cv2
import numpy as np
from PIL import Image, ImageChops

def perform_ela(image_path: str, quality: int = 95, scale_factor: int = 15) -> Image.Image:
    """
    Performs Error Level Analysis (ELA) on an image.
    
    ELA works by saving an image at a specific JPEG quality level and then 
    calculating the absolute difference between the original and the resaved image.
    Since JPEG compression is uniform across a single save, any edited/tampered parts
    will show a higher rate of change (brighter pixels) because they have been resaved
    multiple times or have different original compression statistics.
    
    Parameters:
        image_path (str): Path to the input image file.
        quality (int): JPEG quality level to use for resaving (default is 95).
        scale_factor (int): Multiplier to enhance the visibility of differences (default is 15).
        
    Returns:
        PIL.Image.Image: The enhanced ELA difference image.
    """
    # 1. Open the original image and ensure it is in RGB format
    # (Pillow's convert('RGB') strips any alpha/transparency channels, which JPEGs don't support)
    original = Image.open(image_path).convert("RGB")
    
    # 2. Define a temporary path to save the compressed version
    temp_filename = f"temp_compressed_{os.path.basename(image_path)}.jpg"
    temp_path = os.path.join(os.path.dirname(image_path) or ".", temp_filename)
    
    try:
        # 3. Save the original image as a JPEG with the specified quality level
        original.save(temp_path, "JPEG", quality=quality)
        
        # 4. Re-open the newly saved JPEG image and convert to standard RGB mode
        compressed = Image.open(temp_path).convert("RGB")
        
        # 5. Calculate the absolute pixel-by-pixel difference between original and compressed
        # ImageChops.difference computes |pixel1 - pixel2| for every pixel in the images
        diff = ImageChops.difference(original, compressed)
        
        # 6. Apply scaling to make the subtle differences visible to the human eye.
        # We use Pillow's .point() transform to multiply each pixel value by the scale_factor.
        ela_image = diff.point(lambda p: p * scale_factor)
        
        return ela_image
        
    finally:
        # 7. Clean up the temporary file, ensuring it is deleted even if an exception occurs
        if os.path.exists(temp_path):
            os.remove(temp_path)

def generate_ela_heatmap(ela_image: Image.Image) -> np.ndarray:
    """
    Converts a PIL ELA image into a colorful heatmap using OpenCV's colormap.
    
    Heatmaps are easier to interpret, highlighting areas of high compression mismatch in bright red.
    
    Parameters:
        ela_image (PIL.Image.Image): The ELA difference image.
        
    Returns:
        np.ndarray: The heatmap image in OpenCV BGR format.
    """
    # 1. Convert the PIL Image (RGB) into a NumPy array (OpenCV format)
    # PIL uses RGB, but OpenCV uses BGR, so we convert the channels
    ela_np = np.array(ela_image)
    ela_bgr = cv2.cvtColor(ela_np, cv2.COLOR_RGB2BGR)
    
    # 2. Convert to grayscale to represent intensity (0 to 255)
    gray = cv2.cvtColor(ela_bgr, cv2.COLOR_BGR2GRAY)
    
    # 3. Apply OpenCV's COLORMAP_JET to generate a thermal-style heatmap
    # In JET: Low values are blue, medium are green/yellow, high values are red
    heatmap = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
    
    return heatmap

def detect_suspicious_regions(ela_image: Image.Image, threshold_val: int = 20, min_area: int = 100) -> list:
    """
    Analyzes the ELA image using OpenCV to locate clusters of high-difference pixels.
    These clusters represent localized areas of suspicious modifications.
    
    Parameters:
        ela_image (PIL.Image.Image): The ELA difference image.
        threshold_val (int): Grayscale threshold. Differences above this are suspicious (default 20).
        min_area (int): Minimum size in square pixels for a region to be labeled suspicious (default 100).
        
    Returns:
        list: A list of bounding boxes (dicts) containing the x, y, width, height, and area of suspicious regions.
    """
    # 1. Convert the PIL Image to a grayscale NumPy array
    ela_np = np.array(ela_image)
    ela_gray = cv2.cvtColor(ela_np, cv2.COLOR_RGB2GRAY)
    
    # 2. Threshold the image: set pixels with values >= threshold_val to 255 (white), and others to 0 (black)
    _, thresh = cv2.threshold(ela_gray, threshold_val, 255, cv2.THRESH_BINARY)
    
    # 3. Use morphology (Closing) to merge nearby white pixels into solid shapes
    # This helps group separate dots of noise/edges together into an integrated suspicious region.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    morphed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    # 4. Find external contours of the white shapes in the thresholded image
    contours, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    suspicious_regions = []
    
    # 5. Loop through each contour to filter out tiny noise based on size
    for contour in contours:
        area = cv2.contourArea(contour)
        
        # Only consider regions larger than our minimum area threshold
        if area >= min_area:
            # Get the coordinates and size of the bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            suspicious_regions.append({
                "box": [int(x), int(y), int(w), int(h)],
                "area": float(area)
            })
            
    return suspicious_regions

def draw_suspicious_regions(original_image_path: str, suspicious_regions: list) -> np.ndarray:
    """
    Draws red bounding boxes around detected suspicious regions on the original image.
    
    Parameters:
        original_image_path (str): Path to the original file.
        suspicious_regions (list): List of detected boxes from detect_suspicious_regions.
        
    Returns:
        np.ndarray: The original image with red bounding boxes drawn on it (in BGR format).
    """
    # 1. Read the original image in BGR format
    original = cv2.imread(original_image_path)
    if original is None:
        # If OpenCV fails to read (e.g. if file is PDF), open with PIL and convert
        pil_img = Image.open(original_image_path).convert("RGB")
        original = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        
    # 2. Loop through all regions and draw rectangles
    for region in suspicious_regions:
        x, y, w, h = region["box"]
        # Draw a red rectangle (BGR: 0, 0, 255) with thickness of 2 pixels
        cv2.rectangle(original, (x, y), (x + w, y + h), (0, 0, 255), 2)
        
        # Add a text label indicating "Suspicious" above the rectangle
        cv2.putText(original, "Suspicious", (x, y - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
                    
    return original
