import numpy as np
import cv2 as cv

video1 = cv.VideoCapture('outputSaBELIMIVICNJACIMA.mp4')
video2 = cv.VideoCapture('outputSaPlavomBojom.mp4')

# Get video properties (assuming both videos have the same dimensions)
width = int(video2.get(3))
height = int(video2.get(4))

# Create a VideoWriter object to save the result
result_video = cv.VideoWriter('outputSaBELIMIVICNJACIMAiPLAVIMPUTEM.mp4', cv.VideoWriter_fourcc(*'mp4v'), 100, (width, height))

while True:
    # Read frames from the videos
    ret1, frame = video1.read()
    ret2, frame2 = video2.read()

    # Break the loop if either video ends
    if not ret1 or not ret2:
        break

    # Check if frame dimensions match  ####dodato 14.1.2024.
    if frame.shape[0] != height or frame.shape[1] != width:
        frame = cv.resize(frame, (width, height))

    # Extract blue pixels from the first video
    blue_pixels = np.where(frame2[:, :, 0] == 255)

    # Add blue pixels from the first video to the second video
    frame[blue_pixels] = frame2[blue_pixels]

    # Write the resulting frame to the output video
    result_video.write(frame)

# Release video capture and writer objects
video1.release()
video2.release()
result_video.release()

cv.destroyAllWindows()
