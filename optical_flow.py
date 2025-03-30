import cv2
import argparse
import numpy as np


''' Draws a line on an image with color corresponding to the direction of line
 image im: image to draw line on
 float x, y: starting point of line
 float dx, dy: vector corresponding to line angle and magnitude
'''
def draw_line(im, x, y, dx, dy):
    # Make a copy of the image to avoid modifying the original
    result = im.copy()
    
    # Convert to color image if grayscale
    if len(result.shape) == 2:
        result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
    
    # Convert to uint8 if needed for drawing
    if result.dtype != np.uint8:
        result = np.clip(result, 0, 255).astype(np.uint8)
    
    # Compute magnitude and angle
    mag = np.sqrt(dx*dx + dy*dy)
    angle = np.arctan2(dy, dx)
    
    # HSV color mapping
    # Map angle from [-pi, pi] to [0, 180] for hue
    hue = int((angle + np.pi) * 180 / (2 * np.pi))
    
    # Adjust saturation and value based on magnitude
    saturation = 255
    value = min(int(mag * 10)+80, 255)  # Scale magnitude to brightness
    
    # Create color in HSV and convert to BGR
    color_hsv = np.uint8([[[hue, saturation, value]]])
    color = cv2.cvtColor(color_hsv, cv2.COLOR_HSV2BGR)[0, 0]
    
    # Convert color to tuple for OpenCV
    color = (int(color[0]), int(color[1]), int(color[2]))
    
    # Determine number of dots based on line length
    length = np.hypot(dx, dy)
    num_dots = max(int(length // 1), 1)
    
    # Draw dots along the line with color variation
    for i in range(num_dots + 1):
        # Interpolate position along the line
        dot_x = int(x + (dx * i) / num_dots)
        dot_y = int(y + (dy * i) / num_dots)
        
        # Draw dot with computed color
        cv2.circle(result, (dot_x, dot_y), 1, color, -1)
    
    return result


''' Make an integral image or summed area table from an image
 image im: image to process
 returns: image I such that I[x,y] = sum{i<=x, j<=y}(im[i,j])
'''
def make_integral_image(im):
  im = np.asarray(im, dtype=np.float32)
    
  integral_image = np.zeros((im.shape[0] + 1, im.shape[1] + 1), dtype=np.float32)
    
  for i in range(1, integral_image.shape[0]):
      for j in range(1, integral_image.shape[1]):
          integral_image[i, j] = im[i - 1, j - 1] + integral_image[i - 1, j] + integral_image[i, j - 1] - integral_image[i - 1, j - 1]
    
  return integral_image[1:, 1:]

''' Apply a box filter to an image using an integral image for speed
 image im: image to smooth
 int s: window size for box filter
 returns: smoothed image
'''
def box_filter_image(im,s):
  h, w = im.shape
    
  # Create integral image
  integral = make_integral_image(im)
  
  # Initialize output image
  smoothed = np.zeros_like(im)
  
  # Half window size (for centering the window)
  half_s = s // 2
  
  # Apply box filter using integral image
  for y in range(h):
      for x in range(w):
          # Calculate box boundaries (handle image borders)
          x1 = max(0, x - half_s)
          y1 = max(0, y - half_s)
          x2 = min(w - 1, x + half_s)
          y2 = min(h - 1, y + half_s)
          
          # Calculate sum using integral image
          # For region (x1,y1) to (x2,y2), we need:
          # I[y2,x2] - I[y1-1,x2] - I[y2,x1-1] + I[y1-1,x1-1]
          # With bounds checking:
          sum_region = integral[y2, x2]
          
          if y1 > 0:
              sum_region -= integral[y1-1, x2]
          
          if x1 > 0:
              sum_region -= integral[y2, x1-1]
          
          if x1 > 0 and y1 > 0:
              sum_region += integral[y1-1, x1-1]
          
          # Calculate area of the box (might be smaller at borders)
          area = (y2 - y1 + 1) * (x2 - x1 + 1)
          
          # Compute average
          smoothed[y, x] = sum_region / area
  
  return smoothed


''' Calculate the time-structure matrix of an image pair.
 image im: the input image.
 image prev: the previous image in sequence.
 int s: window size for smoothing.
 returns: structure matrix. 1st channel is Ix^2, 2nd channel is Iy^2,
          3rd channel is IxIy, 4th channel is IxIt, 5th channel is IyIt.
'''
def time_structure_matrix(im,prev,s):
  h, w = im.shape
    
  # Initialize the structure tensor (5 channels)
  S = np.zeros((h, w, 5), dtype=np.float32)
  
  # Calculate image derivatives
  
  # Spatial derivatives using simple difference method
  # Ix - derivative in x direction
  
  Ix = np.zeros_like(im)
  Ix[:, 1:w-1] = (im[:, 2:w] - im[:, 0:w-2]) / 2
  
  # Iy - derivative in y direction
  Iy = np.zeros_like(im)
  Iy[1:h-1, :] = (im[2:h, :] - im[0:h-2, :]) / 2
  
  
  # Temporal derivative It (difference between current and previous frame)
  It = im - prev
  
  # Calculate products for the structure tensor
  Ix2 = Ix * Ix      # Ix^2
  Iy2 = Iy * Iy      # Iy^2
  Ixy = Ix * Iy      # IxIy
  Ixt = Ix * It      # IxIt
  Iyt = Iy * It      # IyIt
  
  # Store the products in the structure tensor
  S[:, :, 0] = Ix2
  S[:, :, 1] = Iy2
  S[:, :, 2] = Ixy
  S[:, :, 3] = Ixt
  S[:, :, 4] = Iyt
  
  # Apply box filter smoothing to each channel of the structure tensor
  for i in range(5):
      S[:, :, i] = box_filter_image(S[:, :, i], s)
  
  return S



'''
Calculate the velocity given a structure image
image S: time-structure image
int stride: 
'''
def velocity_image(S,stride):
  h, w = S.shape[0], S.shape[1]
  V = np.zeros((h, w, 2), dtype=np.float32)
  
  # Loop through image with given stride
  for y in range(0, h, stride):
      for x in range(0, w, stride):
          # Extract components from the structure tensor
          Ix2 = S[y, x, 0]
          Iy2 = S[y, x, 1]
          Ixy = S[y, x, 2]
          Ixt = S[y, x, 3]
          Iyt = S[y, x, 4]
          
          # Small epsilon to avoid division by very small values
          epsilon = 1e-8
          
          # Check for aperture problem cases
          if Ix2 > epsilon and Iy2 <= epsilon:
              # Only horizontal gradient - can only determine x-component
              V[y, x, 0] = -Ixt / Ix2
              V[y, x, 1] = 0
          elif Iy2 > epsilon and Ix2 <= epsilon:
              # Only vertical gradient - can only determine y-component
              V[y, x, 0] = 0
              V[y, x, 1] = -Iyt / Iy2
          else:
              # Try to solve the full system
              det = Ix2 * Iy2 - Ixy * Ixy
              
              if abs(det) > epsilon:
                  # Calculate inverse manually
                  inv_det = 1.0 / det
                  M_inv = np.array([[Iy2 * inv_det, -Ixy * inv_det],
                                    [-Ixy * inv_det, Ix2 * inv_det]])
                  
                  # Compute v = M_inv * b
                  V[y, x, 0] = M_inv[0, 0] * (-Ixt) + M_inv[0, 1] * (-Iyt)
                  V[y, x, 1] = M_inv[1, 0] * (-Ixt) + M_inv[1, 1] * (-Iyt)
              else:
                  # Matrix is not invertible and doesn't fall into aperture problem cases
                  V[y, x, 0] = 0
                  V[y, x, 1] = 0
  
  # If stride > 1, fill in the gaps with interpolation
  if stride > 1:
      # Simple nearest-neighbor interpolation
      for y in range(h):
          for x in range(w):
              if y % stride != 0 or x % stride != 0:
                  grid_y = (y // stride) * stride
                  grid_x = (x // stride) * stride
                  grid_y = min(grid_y, h - 1)
                  grid_x = min(grid_x, w - 1)
                  V[y, x, 0] = V[grid_y, grid_x, 0]
                  V[y, x, 1] = V[grid_y, grid_x, 1]
  
  return V




'''
Draw lines on an image given the velocity
image im: image to draw on
image v: velocity of each pixel
float scale: scalar to multiply velocity by for drawing
'''
def draw_flow(im,v,scale=1.0):
  # Create a copy of the input image for drawing
    if len(im.shape) == 2:  # If grayscale, convert to RGB
        result = np.dstack([im, im, im])
    else:
        result = im.copy()
    
    # Ensure the result is in the correct format for drawing
    result = result.astype(np.float32)
    if np.max(result) <= 1.0:
        result = result * 255.0
    
    # Get dimensions
    h, w = im.shape[0], im.shape[1]
    
    # Draw flow lines
    step = 16  # Sample flow vectors at this interval for clearer visualization
    for y in range(step//2, h, step):
        for x in range(step//2, w, step):
            # Get velocity at this point
            dx = v[y, x, 0] * scale
            dy = v[y, x, 1] * scale
            
            # Skip drawing if motion is too small
            #if dx*dx + dy*dy > 0.1:
                #continue
            
            result = draw_line(result, x, y, dx, dy)
    
    # Normalize the result for display
    
    
    return result





'''
Constrain the absolute value of each image pixel
image im: image to constrain
float v: each pixel will be in range [-v, v]
'''
def constrain_image(im, v):
    # Convert image to numpy array if it's not already
    im = np.asarray(im, dtype=np.float32)
    
    # Clip the values to be in range [-v, v]
    constrained = np.clip(im, -v, v)
    
    return constrained



'''
Calculate the optical flow between two images
image im: current image
image prev: previous image
int smooth: amount to smooth structure matrix by
int stride: downsampling for velocity matrix
returns: velocity matrix
'''
def optical_flow_images(im, prev, smooth=3, stride=1):
    # Convert images to grayscale if they're not already
    if len(im.shape) == 3:
        im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    else:
        im_gray = im.copy()
        
    if len(prev.shape) == 3:
        prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    else:
        prev_gray = prev.copy()
    
    # Convert to float32 for calculations
    im_gray = im_gray.astype(np.float32)
    prev_gray = prev_gray.astype(np.float32)
    
    # Normalize images if they're in [0, 255] range
    if np.max(im_gray) > 1.0:
        im_gray /= 255.0
    if np.max(prev_gray) > 1.0:
        prev_gray /= 255.0
    
    # Calculate the time-structure matrix
    S = time_structure_matrix(im_gray, prev_gray, smooth)
    
    # Calculate the velocity image based on the structure matrix
    V = velocity_image(S, stride)
    
    V = constrain_image(V,255)
    return V



'''
Run optical flow demo on webcam
int smooth: amount to smooth structure matrix by
int stride: downsampling for velocity matrix
int div: downsampling factor for images from webcam
'''
def optical_flow_webcam(smooth=3, stride=1, div=1):
    # Open the webcam
    cap = cv2.VideoCapture(0)
    
    # Check if webcam opened successfully
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    # Read the first frame
    ret, prev_frame = cap.read()
    if not ret:
        print("Error: Could not read from webcam.")
        cap.release()
        return
    
    # Downsample the first frame if div > 1
    if div > 1:
        h, w = prev_frame.shape[0] // div, prev_frame.shape[1] // div
        prev_frame = cv2.resize(prev_frame, (w, h))
    
    # Convert first frame to grayscale
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    
    # Create windows for display
    cv2.namedWindow('Webcam', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Optical Flow', cv2.WINDOW_NORMAL)
    
    print("Running optical flow demo. Press 'q' to quit.")
    
    while True:
        # Read the next frame
        ret, current_frame = cap.read()
        current_frame = cv2.flip(current_frame,1)
        if not ret:
            print("Error: Failed to capture frame.")
            break
        
        # Downsample the current frame if div > 1
        if div > 1:
            h, w = current_frame.shape[0] // div, current_frame.shape[1] // div
            current_frame = cv2.resize(current_frame, (w, h))
        
        # Convert current frame to grayscale
        current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate optical flow
        flow = optical_flow_images(current_gray, prev_gray, smooth=smooth, stride=stride)
        
        # Visualize the flow
        flow_img = draw_flow(current_frame, flow, scale=5.0)
        
        # Display the frames
        cv2.imshow('Webcam', current_frame)
        cv2.imshow('Optical Flow', flow_img)
        
        # Update previous frame
        prev_gray = current_gray.copy()
        
        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    print("Optical flow demo ended.")

def __main__():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Optical Flow Visualization")
    
    # Create mutually exclusive group for webcam vs image modes
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument('--webcam', action='store_true', help="Use webcam for real-time optical flow")
    mode_group.add_argument('--image1', type=str, help="Path to the first input image")
    
    # Optional arguments
    parser.add_argument('--image2', type=str, help="Path to the second input image (required with --image1)")
    parser.add_argument('--output', type=str, help="Path for the output image (required with --image1)")
    parser.add_argument('--smooth', type=int, default=3, help="Window size for smoothing")
    parser.add_argument('--stride', type=int, default=1, help="Stride for velocity calculation")
    parser.add_argument('--div', type=int, default=1, help="Downsampling factor for images")
    
    args = parser.parse_args()
    
    # Webcam mode
    if args.webcam:
        optical_flow_webcam(smooth=args.smooth, stride=args.stride, div=args.div)
    
    # Image processing mode
    else:
        # Check if required arguments for image mode are provided
        if not args.image2 or not args.output:
            parser.error("--image1 requires --image2 and --output")
            
        # Load the images
        img1 = cv2.imread(args.image1)
        img2 = cv2.imread(args.image2)
        
        if img1 is None or img2 is None:
            print("Error: Could not load one or both images.")
            return
            
        # Calculate optical flow
        flow = optical_flow_images(img1, img2, smooth=args.smooth, stride=args.stride)
        
        # Visualize and save the result
        flow_img = draw_flow(img1, flow, scale=5.0)
        cv2.imwrite(args.output, flow_img)
        print(f"Optical flow result saved to {args.output}")

if __name__ == "__main__":
    __main__()
