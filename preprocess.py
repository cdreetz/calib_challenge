import cv2
import numpy as np

def calculate_optical_flow(video_path):
    # Open the video file
    video = cv2.VideoCapture(video_path)

    # Check if the video file opened successfully
    if not video.isOpened():
        print("Error opening video file:", video_path)
        return

    # Read the first frame
    ret, prev_frame = video.read()

    # Create an empty mask for visualizing the optical flow
    mask = np.zeros_like(prev_frame)

    # Convert the first frame to grayscale
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    # Create a window to display the optical flow
    cv2.namedWindow("Optical Flow")

    # Loop through the video frames
    while True:
        # Read the next frame
        ret, curr_frame = video.read()

        if not ret:
            break

        # Convert the current frame to grayscale
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

        # Create initial points for optical flow
        prevPts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)

        # Calculate the optical flow using Lucas-Kanade method
        optical_flow, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prevPts, None)

        # Select good points for optical flow visualization
        good_old = prevPts[status == 1]
        good_new = optical_flow[status == 1]

        # Draw the optical flow vectors on the mask
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel().astype(int)
            c, d = old.ravel().astype(int)
            mask = cv2.line(mask, (a, b), (c, d), (0, 255, 0), 2)
            curr_frame = cv2.circle(curr_frame, (a, b), 5, (0, 255, 0), -1)

        # Overlay the mask on the current frame
        output_frame = cv2.add(curr_frame, mask)

        # Display the resulting frame
        cv2.imshow("Optical Flow", output_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Update the previous frame and grayscale image
        prev_frame = curr_frame
        prev_gray = curr_gray

    # Release the video capture object and close any open windows
    video.release()
    cv2.destroyAllWindows()

# List of video file names
video_files = ['0gray.mp4', '1gray.mp4', '2gray.mp4', '3gray.mp4']

# Iterate over the video files
for video_file in video_files:
    # Generate the full video file path
    video_path = r"C://Users/Christian/Desktop/calib_challenge/calib_challenge/labeled/train/" + video_file

    # Call the calculate_optical_flow function with the current video file
    calculate_optical_flow(video_path)
