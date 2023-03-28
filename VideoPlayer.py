import sys
import os
import argparse

from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QLabel, QPushButton, QSlider, QAction, QMenu, QMessageBox
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap

import cv2
import numpy as np
import matplotlib.pyplot as plt

from VideoPlayerLibrary import VideoPlayerLibrary

class VideoPlayer(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Video Player")
        self.resize(800, 600)

        # Set up the video player library
        self.video_player_library = None

        # Set up the UI elements
        self.video_label = QLabel()
        self.setCentralWidget(self.video_label)

        self.play_button = QPushButton("Play")
        self.pause_button = QPushButton("Pause")
        self.stop_button = QPushButton("Stop")
        self.previous_button = QPushButton("Previous")
        self.next_button = QPushButton("Next")

        self.play_button.clicked.connect(self.play_video)
        self.pause_button.clicked.connect(self.pause_video)
        self.stop_button.clicked.connect(self.stop_video)
        self.previous_button.clicked.connect(self.previous_frame)
        self.next_button.clicked.connect(self.next_frame)

        toolbar = self.addToolBar("Controls")
        toolbar.addWidget(self.play_button)
        toolbar.addWidget(self.pause_button)
        toolbar.addWidget(self.stop_button)
        toolbar.addWidget(self.previous_button)
        toolbar.addWidget(self.next_button)

        self.statusBar()

        # Set up the timer
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)

        # Set up the video playback state
        self.playing = False
        self.paused = False

        # Set up the menu bar
        menubar = self.menuBar()

        # Set up the file menu
        file_menu = menubar.addMenu("File")

        open_file_action = QAction("Open File", self)
        open_file_action.setShortcut("Ctrl+O")
        open_file_action.triggered.connect(self.open_file)
        file_menu.addAction(open_file_action)

        close_file_action = QAction("Close File", self)
        close_file_action.setShortcut("Ctrl+W")
        close_file_action.triggered.connect(self.close_file)
        file_menu.addAction(close_file_action)

        file_menu.addSeparator()

        exit_action = QAction("Exit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # Set up the playback menu
        playback_menu = menubar.addMenu("Playback")

        play_action = QAction("Play", self)
        play_action.setShortcut("Space")
        play_action.triggered.connect(self.play_video)
        playback_menu.addAction(play_action)

        pause_action = QAction("Pause", self)
        pause_action.setShortcut("Space")
        pause_action.triggered.connect(self.pause_video)
        playback_menu.addAction(pause_action)

        stop_action = QAction("Stop", self)
        stop_action.setShortcut("Ctrl+Space")
        stop_action.triggered.connect(self.stop_video)
        playback_menu.addAction(stop_action)

        playback_menu.addSeparator()

        previous_frame_action = QAction("Previous Frame", self)
        previous_frame_action.setShortcut("Left")
        previous_frame_action.triggered.connect(self.previous_frame)
        playback_menu.addAction(previous_frame_action)

        next_frame_action = QAction("Next Frame", self)
        next_frame_action.setShortcut("Right")
        next_frame_action.triggered.connect(self.next_frame)
        playback_menu.addAction(next_frame_action)

        # Set up the filter menu
        filter_menu = menu_bar.addMenu("Filter")

        grayscale_filter_action = QAction("Grayscale", self)
        grayscale_filter_action.triggered.connect(self.apply_grayscale_filter)

        negative_filter_action = QAction("Negative", self)
        negative_filter_action.triggered.connect(self.apply_negative_filter)

        filter_menu.addAction(grayscale_filter_action)
        filter_menu.addAction(negative_filter_action)

        # Set up the histogram menu
        histogram_menu = menu_bar.addMenu("Histogram")

        frame_histogram_action = QAction("Frame Histogram", self)
        frame_histogram_action.triggered.connect(self.show_frame_histogram)

        video_histogram_action = QAction("Video Histogram", self)
        video_histogram_action.triggered.connect(self.show_video_histogram)

        histogram_menu.addAction(frame_histogram_action)
        histogram_menu.addAction(video_histogram_action)

        # Set up the playback slider
        self.playback_slider = QSlider(Qt.Horizontal)
        self.playback_slider.setFocusPolicy(Qt.NoFocus)
        self.playback_slider.sliderMoved.connect(self.set_slider_position)

        self.statusBar().addPermanentWidget(self.playback_slider)

        # Set up the timer
        self.timer = QTimer(self)
        self.timer.setInterval(33)  # 30 frames per second
        self.timer.timeout.connect(self.update_frame)

        # Set up the video playback state
        self.playing = False
        self.paused = False
        self.update_ui()

    def open_file_dialog(self):
        file_dialog = QFileDialog(self)
        file_dialog.setFileMode(QFileDialog.ExistingFile)
        file_dialog.setNameFilter("Video files (*.mp4 *.avi *.mov)")
        if file_dialog.exec_():
            file_paths = file_dialog.selectedFiles()
            if file_paths:
                self.open_video(file_paths[0])

    def open_video(self, video_path):
        self.video_player = VideoPlayerLibrary(video_path)
        self.update_ui()
        self.playback_slider.setRange(0, self.video_player.get_total_frames()-1)
        self.playback_slider.setValue(0)
        self.playback_slider.setEnabled(True)

    def close_video(self):
        self.video_player = None
        self.update_ui()

def update_ui(self):
        if self.video_player is None:
            self.setWindowTitle("Video Player")
            self.setGeometry(100, 100, 300, 50)
            self.play_button.setEnabled(False)
            self.pause_button.setEnabled(False)
            self.stop_button.setEnabled(False)
            self.previous_button.setEnabled(False)
            self.next_button.setEnabled(False)
            self.playback_slider.setEnabled(False)
            self.save_frame_button.setEnabled(False)
            self.filter_menu.setEnabled(False)
            self.histogram_menu.setEnabled(False)
        else:
            title = f"Video Player - {os.path.basename(self.video_player.get_video_path())}"
            self.setWindowTitle(title)
            self.setGeometry(100, 100, 800, 600)
            self.play_button.setEnabled(True)
            self.pause_button.setEnabled(True)
            self.stop_button.setEnabled(True)
            self.previous_button.setEnabled(True)
            self.next_button.setEnabled(True)
            self.playback_slider.setEnabled(True)
            self.save_frame_button.setEnabled(True)
            self.filter_menu.setEnabled(True)
            self.histogram_menu.setEnabled(True)

            frame = self.video_player.get_frame()
            if frame is not None:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = QImage(frame.data, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
                frame = QPixmap.fromImage(frame)
                self.video_label.setPixmap(frame)
                self.playback_slider.setMaximum(self.video_player.get_total_frames() - 1)
                self.playback_slider.setValue(self.video_player.get_frame_index())

def playback_slider_changed(self, value):
    if self.video_player is not None:
        frame_index = self.playback_slider.value()
        self.video_player.seek_frame(frame_index)
        self.update_ui()

def apply_filter(self, filter_func):
    if self.video_player is not None:
        self.video_player.apply_filter(filter_func)
        self.update_ui()

def show_frame_histogram(self):
    if self.video_player is not None:
        frame_index = self.video_player.get_frame_index()
        histogram = self.video_player.get_frame_histogram(frame_index)
        if histogram is not None:
            plt.plot(histogram)
            plt.xlim([0, 256])
            plt.title("Frame Histogram")
            plt.show()

def show_video_histogram(self):
    if self.video_player is not None:
        histogram = self.video_player.get_video_histogram()
        if histogram is not None:
            plt.plot(histogram)
            plt.xlim([0, 256])
            plt.title("Video Histogram")
            plt.show()

def save_frame(self):
    if self.video_player is not None:
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Frame As", "", "PNG Image (*.png)")
        if file_path:
            self.video_player.save_frame(self.video_player.get_frame_index(), file_path)

def open_video(self):
    file_path, _ = QFileDialog.getOpenFileName(self, "Open Video File", "", "Video Files (*.mp4 *.avi)")
    if file_path:
        self.video_player = VideoPlayerLibrary(file_path)
        self.playback_slider.setRange(0, self.video_player.get_total_frames() - 1)
        self.playback_slider.setValue(0)
        self.update_ui()
        self.playback_slider.setEnabled(True)

def close_video(self):
    self.video_player = None
    self.playback_slider.setValue(0)
    self.playback_slider.setEnabled(False)
    self.update_ui()

def play_video(self):
    if self.video_player is not None:
        self.video_player.start_playback()
        self.playback_slider.setValue(self.video_player.get_frame_index())

def pause_video(self):
    if self.video_player is not None:
        self.video_player.pause_playback()

def stop_video(self):
    if self.video_player is not None:
        self.video_player.stop_playback()
        self.playback_slider.setValue(self.video_player.get_frame_index())

def previous_frame(self):
    if self.video_player is not None:
        current_frame_index = self.video_player.get_frame_index()
        if current_frame_index > 0:
            self.video_player.seek_frame(current_frame_index - 1)
            self.update_ui()

def next_frame(self):
    if self.video_player is not None:
        current_frame_index = self.video_player.get_frame_index()
        total_frames = self.video_player.get_total_frames()
        if current_frame_index < total_frames - 1:
            self.video_player.seek_frame(current_frame_index + 1)
            self.update_ui()

def play_pause_video(self):
    if self.video_player is not None:
        if self.video_player.is_playing():
            self.video_player.pause_playback()
        else:
            self.video_player.start_playback()
        self.playback_slider.setValue(self.video_player.get_frame_index())

def set_playback_speed(self, speed):
    if self.video_player is not None:
        self.video_player.set_playback_speed(speed)

def set_brightness(self, brightness):
    if self.video_player is not None:
        self.video_player.set_brightness(brightness)

def set_contrast(self, contrast):
    if self.video_player is not None:
        self.video_player.set_contrast(contrast)

    def set_saturation(self, saturation):
        if self.video_player is not None:
            def saturation_filter(frame):
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                h, s, v = cv2.split(hsv)
                s = cv2.addWeighted(s, saturation, np.zeros_like(s), 1 - saturation, 0)
                hsv = cv2.merge([h, s, v])
                frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
                return frame

            self.video_player.apply_filter(saturation_filter)
            self.update_ui()
