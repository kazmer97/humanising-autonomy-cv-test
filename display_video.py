from math import floor
from typing import NoReturn
import load_video_detections

import cv2


def open_video(path: str) -> cv2.VideoCapture:
    """Opens a video file.

    Args:
        path: the location of the video file to be opened

    Returns:
        An opencv video capture file.
    """
    video_capture = cv2.VideoCapture(path)
    if not video_capture.isOpened():
        raise RuntimeError(f'Video at "{path}" cannot be opened.')
    return video_capture


# def get_frame_dimensions(video_capture: cv2.VideoCapture) -> tuple[int, int]:
def get_frame_dimensions(video_capture: cv2.VideoCapture) -> tuple():
    """Returns the frame dimension of the given video.

    Args:
        video_capture: an opencv video capture file.

    Returns:
        A tuple containing the height and width of the video frames.

    """
    return (video_capture.get(cv2.CAP_PROP_FRAME_WIDTH), video_capture.get(
        cv2.CAP_PROP_FRAME_HEIGHT
    ))


def get_frame_display_time(video_capture: cv2.VideoCapture) -> int:
    """Returns the number of milliseconds each frame of a VideoCapture should be displayed.

    Args:
        video_capture: an opencv video capture file.

    Returns:
        The number of milliseconds each frame should be displayed for.
    """
    frames_per_second = video_capture.get(cv2.CAP_PROP_FPS)
    return floor(1000 / frames_per_second)


def is_window_open(title: str) -> bool:
    """Checks to see if a window with the specified title is open."""

    # all attempts to get a window property return -1 if the window is closed
    return cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) >= 1


def main(video_path: str, resource_path: str, title: str) -> NoReturn:
    """Displays a video at half size until it is complete or the 'q' key is pressed.

    Args:
        video_path: the location of the video to be displayed
        title: the title to display in the video window
    """

    video_capture = open_video(video_path)
    width, height = get_frame_dimensions(video_capture)
    wait_time = get_frame_display_time(video_capture)

    video_boxes = load_video_detections.load_bounding_boxes(resource_path)

    tagged_video_boxes = load_video_detections.assign_id(video_boxes)

    try:
        # read the first frame
        success, frame = video_capture.read()

        # create the window
        cv2.namedWindow(title, cv2.WINDOW_AUTOSIZE)

        frame_counter = 1

        # run whilst there are frames and the window is still open
        # had to change this line to get it to work on mac
        while success:
            
            # add rectangles, centre point and ID text to the video
            for b in tagged_video_boxes[frame_counter]:
                cv2.rectangle(frame, pt1= (b.x, b.y), pt2=(b.x + b.width, b.y + b.height), color = b.color)
                cv2.circle(frame, center=b.centre_point(), radius=0, color=(0,0,255), thickness=10)
                cv2.putText(frame, str(b.id), b.centre_point(), fontFace=1, fontScale= 4, color=(255,0,0), thickness=3)

            # shrink it
            smaller_image = cv2.resize(frame, (floor(width // 2), floor(height // 2)))

            # display it
            cv2.imshow(title, smaller_image)

            
            # slight change to allow it to work on mac
            if cv2.waitKey(wait_time) == ord("q"):
                break

            # read the next frame
            frame_counter += 1
            success, frame = video_capture.read()
    finally:
        video_capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    VIDEO_PATH = "resources/video_1.mp4"
    RESOURCE_PATH = "resources/video_1_detections.json"
    main(VIDEO_PATH,RESOURCE_PATH, "My Video")
