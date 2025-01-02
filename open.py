import cv2

def check_camera_accessibility(camera_index):
    """
    Check if a camera is accessible and can be opened for video capture.
    
    Args:
        camera_index (int): The index of the camera to check (typically 0 for built-in webcam, 1 for external camera, etc.)
    
    Returns:
        bool: True if the camera is accessible, False otherwise
    """
    try:
        # Attempt to open the camera
        cap = cv2.VideoCapture(camera_index)
        
        # Check if the camera opened successfully
        if not cap.isOpened():
            print(f"Camera {camera_index} could not be opened.")
            return False
        
        # Try to read a frame
        ret, frame = cap.read()
        
        # Release the camera
        cap.release()
        
        # Check if a frame was successfully read
        if not ret:
            print(f"Unable to capture a frame from camera {camera_index}.")
            return False
        
        print(f"Camera {camera_index} is accessible and working correctly.")
        return True
    
    except Exception as e:
        print(f"An error occurred while checking camera {camera_index}: {e}")
        return False

def list_available_cameras(max_cameras=10):
    """
    Scan through possible camera indices to find accessible cameras.
    
    Args:
        max_cameras (int): Maximum number of camera indices to check (default is 10)
    
    Returns:
        list: List of camera indices that are accessible
    """
    available_cameras = []
    
    for i in range(max_cameras):
        if check_camera_accessibility(i):
            available_cameras.append(i)
    
    return available_cameras

# Example usage

# Check a specific camera index
print("Checking camera 0:")
is_camera_0_available = check_camera_accessibility(0)

# List all available cameras
print("\nListing available cameras:")
available_cameras = list_available_cameras()
print(f"Available camera indices: {available_cameras}")