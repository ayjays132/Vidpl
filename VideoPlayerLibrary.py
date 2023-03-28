import cv2
import numpy as np
import time

class VideoPlayerLibrary:
    
    def __init__(self, video_path):
        self.video_path = video_path
        self.video_capture = cv2.VideoCapture(video_path)
        self.current_frame = None
        self.current_frame_index = 0
        self.total_frames = int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.video_capture.get(cv2.CAP_PROP_FPS)
        self.frame_width = int(self.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.playing = False
        self.delay = int(1000 / self.fps)
    
    def __del__(self):
        self.video_capture.release()
    
    def read_frame(self):
        ret, frame = self.video_capture.read()
        if ret:
            self.current_frame = frame
            self.current_frame_index = int(self.video_capture.get(cv2.CAP_PROP_POS_FRAMES))
            return True
        else:
            return False
    
    def seek_frame(self, frame_index):
        if frame_index < 0:
            frame_index = 0
        elif frame_index >= self.total_frames:
            frame_index = self.total_frames - 1
        self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        self.read_frame()
    
    def get_frame(self, frame_index=None):
        if frame_index is None:
            return self.current_frame
        else:
            if frame_index < 0 or frame_index >= self.total_frames:
                return None
            self.seek_frame(frame_index)
            return self.current_frame
    
    def get_frame_index(self):
        return self.current_frame_index
    
    def get_total_frames(self):
        return self.total_frames
    
    def get_fps(self):
        return self.fps
    
    def get_frame_size(self):
        return (self.frame_width, self.frame_height)
    
    def start_playback(self):
        self.playing = True
        while self.playing:
            if self.read_frame():
                cv2.imshow('Video Player', self.current_frame)
            else:
                self.stop_playback()
            if cv2.waitKey(self.delay) & 0xFF == ord('q'):
                self.stop_playback()
        cv2.destroyAllWindows()
    
    def stop_playback(self):
        self.playing = False
    
    def save_frame(self, frame_index, file_path):
        frame = self.get_frame(frame_index)
        if frame is not None:
            cv2.imwrite(file_path, frame)
    
    def get_frame_histogram(self, frame_index):
        frame = self.get_frame(frame_index)
        if frame is None:
            return None
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        histogram = cv2.calcHist([gray_frame], [0], None, [256], [0, 256])
        return histogram
    
    def get_video_histogram(self):
        histogram = np.zeros((256,))
        for i in range(self.total_frames):
            frame_histogram = self.get_frame_histogram(i)
            histogram += frame_histogram.flatten()
        return histogram
    
    def apply_filter(self, filter_func):
        for i in range(self.total_frames):
            frame = self.get_frame(i)
            filtered_frame = filter_func(frame)
            self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, i)
            self.video_capture.write(filtered_frame)
        self.video_capture.release()

    def invert_colors(self, frame):
        return cv2.bitwise_not(frame)

    def mirror_horizontal(self, frame):
        return cv2.flip(frame, 1)

    def mirror_vertical(self, frame):
        return cv2.flip(frame, 0)

    def apply_gray_scale(self, frame):
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    def apply_canny_edge_detection(self, frame, threshold1=100, threshold2=200):
        edges = cv2.Canny(frame, threshold1, threshold2)
        return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    def apply_sobel_edge_detection(self, frame, dx=1, dy=0, ksize=3, scale=1, delta=0):
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        sobel_x = cv2.Sobel(gray_frame, cv2.CV_64F, dx, 0, ksize=ksize, scale=scale, delta=delta)
        sobel_y = cv2.Sobel(gray_frame, cv2.CV_64F, 0, dy, ksize=ksize, scale=scale, delta=delta)
        abs_sobel_x = cv2.convertScaleAbs(sobel_x)
        abs_sobel_y = cv2.convertScaleAbs(sobel_y)
        sobel = cv2.addWeighted(abs_sobel_x, 0.5, abs_sobel_y, 0.5, 0)
        return cv2.cvtColor(sobel, cv2.COLOR_GRAY2BGR)

    def apply_harris_corner_detection(self, frame, block_size=2, ksize=3, k=0.04):
        gray_frame = np.float32(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        corners = cv2.cornerHarris(gray_frame, block_size, ksize, k)
        corners = cv2.dilate(corners, None)
        frame[corners > 0.01 * corners.max()] = [0, 0, 255]
        return frame

    def apply_lucas_kanade_optical_flow(self, frame1, frame2):
        gray_frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray_frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        corners1 = cv2.goodFeaturesToTrack(gray_frame1, maxCorners=100, qualityLevel=0.01, minDistance=10)
        corners2, status, errors = cv2.calcOpticalFlowPyrLK(gray_frame1, gray_frame2, corners1, None)
        for i in range(corners2.shape[0]):
            x1, y1 = corners1[i].ravel()
            x2, y2 = corners2[i].ravel()
            cv2.line(frame2, (x1, y1), (x2, y2), (0, 255, 0), 2)
        return frame2
