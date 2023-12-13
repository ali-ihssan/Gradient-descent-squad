import cv2
import os
import csv
import numpy as np

frame_array = []

# from keras.preprocessing.image import ImageDataGenerator

# # Create an image data generator object
# datagen = ImageDataGenerator(
#     rotation_range=20,
#     width_shift_range=0.2,
#     height_shift_range=0.2,
#     shear_range=0.2,
#     zoom_range=0.2,
#     horizontal_flip=True,
#     fill_mode='nearest')


def normalize_image(image):
    # normalized_image = np.zeros(image.shape)
    # normalized_image = cv2.normalize(
    #     image, normalized_image, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    normalized_image = image/255.0
    return normalized_image


def standardize_image(image):
    mean, std_dev = cv2.meanStdDev(image)
    mean = mean.reshape(-1)
    std_dev = std_dev.reshape(-1)
    return (image - mean) / std_dev


def video_to_frames(video_path, output_root_folder, start_time, end_time, class_name, type, interval_length=3, output_fps=14, size=(640, 480)):
    if not os.path.isfile(video_path):
        print(f"Error: video file {video_path} does not exist.")
        return

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    video_capture = cv2.VideoCapture(video_path)

    if not video_capture.isOpened():
        print(f"Error: could not open video file {video_path}.")
        return

    os.makedirs(output_root_folder, exist_ok=True)
    fps = int(video_capture.get(cv2.CAP_PROP_FPS))
    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    num_intervals = (end_time - start_time) // interval_length

    if num_intervals == 0:
        print(f"Error: interval length {interval_length} is too large.")
        return

    print(f"Processing video: {video_name}")
    print(f"Total frames: {total_frames}")
    print(f"Frames per second: {fps}")
    print(f"Number of intervals: {num_intervals}")
    print(f"Output FPS: {output_fps}")
    print(f"fps // output_fps: {fps // output_fps}")

    frame_skip = fps // output_fps
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or use 'XVID'

    for interval in range(num_intervals):
        start_frame = int((start_time + interval * interval_length) * fps)
        end_frame = int((start_time + (interval + 1) * interval_length) * fps)
        output_folder = os.path.join(
            output_root_folder, class_name, f"{video_name}_start{start_time}_end{end_time}_interval{interval + 1}")
        os.makedirs(output_folder, exist_ok=True)
        output_file = os.path.join(
            output_folder, )

        out = cv2.VideoWriter(output_file, fourcc, output_fps, (int(
            video_capture.get(3)), int(video_capture.get(4))))

        with open(os.path.join(output_root_folder, 'content.csv'), 'a', newline='') as csvfile:
            fieldnames = ['frame_name', "video_name",
                          'interval', 'frame_number', 'class', 'type', 'path']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            if csvfile.tell() == 0:
                writer.writeheader()

            frame_counter = 0
            for frame_number in range(start_frame, end_frame, frame_skip):
                video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                ret, frame = video_capture.read()
                if not ret:
                    print(f"Error reading frame {frame_number}")
                    break
                if ret:
                    out.write(frame)
                    frame_counter += 1
                    if frame_counter <= 41:

                        # frame = normalize_image(frame)
                        frame = cv2.resize(frame, size)

                        # Perform image augmentation
                        # frame = datagen.random_transform(frame)

                        out.write(frame)
                        # Save each frame as an image
                        output_image_file = os.path.join(
                            output_folder, f"{video_name}_{class_name}_frame{frame_number}.png")
                        cv2.imwrite(output_image_file, frame)
                        # Append the frame to the frame_array
                        frame_array.append(frame)
                        # Write the frame information to the CSV file
                        writer.writerow({'frame_name': f"{video_name}_{class_name}_frame{frame_number}", 'video_name': video_name,
                                        'interval': interval, 'frame_number': frame_number, 'class': class_name, 'type': type, 'path': output_image_file})
                        out.release()

    video_capture.release()


def process_videos_in_folder(input_folder, output_root_folder, video_info):
    global frame_array
    if video_info is None:
        return
    if not os.path.exists(output_root_folder):
        os.makedirs(output_root_folder)
    for video_file, start_time, end_time, class_name, type in video_info:

        if end_time - start_time < 3:
            print(
                f"Skipping video {video_file} start: {start_time} end: {end_time} because the interval is less than 3 seconds")
            continue

        video_path = os.path.join(input_folder, video_file)
        video_to_frames(video_path, output_root_folder,
                        start_time, end_time, class_name, type)
    frame_array = np.array(frame_array)
    np.save('./content/frame_array.npy', frame_array)


if __name__ == "__main__":
    # Set the path to the folder containing video files
    input_folder = "./content/Videos"

    # Set the root folder for output frames
    output_root_folder = "./content/outFrames"

    # Define video information as a list of tuples (video_file, start_time, end_time)
    video_info = [
        ("1.mp4", 38, 41, "UD", "Train"),
        ("1.mp4", 64, 68, "UD", "Train"),
        ("1.mp4", 97, 100, "UD", "Train"),
        ("1.mp4", 110, 114, "UD", "Train"),
        ("1.mp4", 135, 138, "UD", "Train"),
        ("1.mp4", 0, 3, "None", "Train"),
        ("1.mp4", 70, 74, "None", "Train"),

        ("3.mp4", 0, 4, "WF", "Test"),
        ("3.mp4", 33, 36, "WF", "Test"),
        ("3.mp4", 42, 45, "WF", "Test"),
        ("3.mp4", 65, 72, "WF", "Test"),
        ("3.mp4", 28, 31, "None", "Test"),
        ("3.mp4", 60, 64, "None", "Test"),


        ("4.mp4", 3, 9, "UD", "Test"),
        ("4.mp4", 10, 14, "None", "Test"),

        ("5.mp4", 4, 12, "WF", "Test"),
        ("5.mp4", 0, 3, "None", "Test"),

        ("6.mp4", 0, 3, "UD", "Train"),
        ("6.mp4", 19, 22, "UD", "Train"),
        ("6.mp4", 41, 44, "None", "Train"),

        ("7.mp4", 5, 8, "UD", "Test"),
        ("7.mp4", 14, 20, "UD", "Test"),
        ("7.mp4", 21, 24, "None", "Test"),

        ("8.mp4", 3, 10, "UD", "Test"),
        ("8.mp4", 13, 16, "UD", "Test"),
        ("8.mp4", 10, 13, "None", "Test"),

        ("9.mp4", 0, 3, "WF", "Test"),
        ("9.mp4", 18, 24, "WF", "Test"),
        ("9.mp4", 7, 10, "None", "Test"),

        ("10.mp4", 0, 5, "UD", "Train"),
        ("10.mp4", 15, 18, "UD", "Train"),
        ("10.mp4", 69, 73, "UD", "Train"),
        ("10.mp4", 7, 10, "None", "Train"),

        ("12.mp4", 14, 18, "UD", "Train"),
        ("12.mp4", 9, 13, "None", "Train"),

        ("13.mp4", 23, 26, "UD", "Train"),
        ("13.mp4", 0, 3, "WF", "Train"),

        ("15.mp4", 19, 22, "WF", "Train"),
        ("15.mp4", 47, 51, "WF", "Train"),
        ("15.mp4", 70, 75, "WF", "Train"),
        ("15.mp4", 89, 92, "WF", "Train"),
        ("15.mp4", 100, 105, "WF", "Train"),
        ("15.mp4", 123, 127, "WF", "Train"),
        ("15.mp4", 220, 224, "WF", "Train"),
        ("15.mp4", 0, 3, "None", "Train"),
        ("15.mp4", 23, 28, "None", "Train"),
        ("15.mp4", 60, 65, "None", "Train"),
        ("15.mp4", 159, 164, "None", "Train"),

        ("16.mp4", 3, 20, "WF", "Train"),
        ("16.mp4", 43, 50, "WF", "Train"),
        ("16.mp4", 63, 67, "WF", "Train"),
        ("16.mp4", 104, 109, "WF", "Train"),
        ("16.mp4", 36, 40, "None", "Train"),
        ("16.mp4", 50, 55, "None", "Train"),
        ("16.mp4", 60, 63, "None", "Train"),

        ("17.mp4", 4, 6, "WF", "Train"),
        ("17.mp4", 0, 3, "None", "Train"),
        ("17.mp4", 44, 46, "None", "Train"),

        ("18.mp4", 18, 23, "UD", "Train"),
        ("18.mp4", 33, 37, "UD", "Train"),
        ("18.mp4", 2, 5, "None", "Train"),
        ("18.mp4", 8, 11, "None", "Train"),

        ("19.mp4", 32, 40, "UD", "Train"),
        ("19.mp4", 44, 48, "UD", "Train"),
        ("19.mp4", 11, 14, "None", "Train"),

        ("20.mp4", 3, 8, "WF", "Train"),
        ("20.mp4", 28, 30, "WF", "Train"),
        ("20.mp4", 9, 15, "None", "Train"),
        ("20.mp4", 31, 35, "None", "Train"),


        ("21.mp4", 175, 190, "UD", "Train"),
        ("21.mp4", 205, 219, "UD", "Train"),
        ("21.mp4", 40, 44, "None", "Train"),
        ("21.mp4", 240, 244, "None", "Train"),

        ("22.mp4", 94, 98, "UD", "Train"),
        ("22.mp4", 0, 3, "None", "Train"),
        ("22.mp4", 15, 18, "None", "Train"),
        ("22.mp4", 60, 64, "None", "Train"),

        ("23.mp4", 6, 9, "WF", "Train"),
        ("23.mp4", 3, 6, "None", "Train"),
        ("23.mp4", 19, 22, "None", "Train"),

        ("24.mp4", 2, 5, "WF", "Train"),
        ("24.mp4", 41, 45, "WF", "Train"),
        ("24.mp4", 60, 66, "WF", "Train"),
        ("24.mp4", 94, 99, "WF", "Train"),
        ("24.mp4", 6, 10, "None", "Train"),
        ("24.mp4", 26, 30, "None", "Train"),

        ("25.mp4", 8, 15, "WF", "Train"),
        ("25.mp4", 15, 18, "None", "Train"),

        ("26.mp4", 4, 8, "WF", "Train"),
        ("26.mp4", 11, 15, "None", "Train"),


        # Add more videos with their respective start and end times
    ]

    # Call the function to process videos in the folder and save frames within each 3-second interval
    process_videos_in_folder(input_folder, output_root_folder, video_info)
