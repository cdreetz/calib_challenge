import cv2

def preprocess_video(video_path):
    # Open the video file
    video = cv2.VideoCapture(video_path)

    # Check if the video file opened successfully
    if not video.isOpened():
        print("Error opening video file:", video_path)
        return

    # Create a VideoWriter object to save the preprocessed frames
    output_path = "preprocessed_video.mp4"
    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video.get(cv2.CAP_PROP_FPS)
    codec = cv2.VideoWriter_fourcc(*"mp4v")
    output_video = cv2.VideoWriter(output_path, codec, fps, (frame_width, frame_height), isColor=False)

    # Read and process each frame of the video
    while video.isOpened():
        ret, frame = video.read()

        if not ret:
            break

        # Convert the frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Write the preprocessed frame to the output video
        output_video.write(gray_frame)

        # Display the preprocessed frame (optional)
        cv2.imshow("Preprocessed Frame", gray_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture and output writer objects
    video.release()
    output_video.release()

    # Close any open windows
    cv2.destroyAllWindows()

    print("Preprocessing complete. Preprocessed video saved as:", output_path)

# Call the preprocess_video function with the path to your .mp4 video file
video_path = "path/to/your/video.mp4"
preprocess_video(video_path)
