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

        # Set up the playback slider
        self.playback_slider = QSlider(Qt.Horizontal)
        self.playback_slider.setTickPosition(QSlider.TicksBothSides)
        self.playback_slider.setTickInterval(10)
        self.playback_slider.setSingleStep(1)
        self.playback_slider.sliderMoved.connect(self.playback_slider_changed)
        self.playback_slider.setEnabled(False)
        self.statusBar().addPermanentWidget(self.playback_slider)

        # Set up the file menu
        file_menu = self.menuBar().addMenu("File")

        open_action = QAction("Open", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self.open_video)

        close_action = QAction("Close", self)
        close_action.setShortcut("Ctrl+W")
        close_action.triggered.connect(self.close_video)

        file_menu.addAction(open_action)
        file_menu.addAction(close_action)

        # Set up the playback menu
    playback_menu = menubar.addMenu("Playback")

    play_action = QAction("Play", self)
    play_action.setShortcut("Space")
    play_action.triggered.connect(self.play_video)
    playback_menu.addAction(play_action)

    pause_action = QAction("Pause", self)
    pause_action.setShortcut("Ctrl+P")
    pause_action.triggered.connect(self.pause_video)
    playback_menu.addAction(pause_action)

    stop_action = QAction("Stop", self)
    stop_action.setShortcut("Ctrl+S")
    stop_action.triggered.connect(self.stop_video)
    playback_menu.addAction(stop_action)

    previous_action = QAction("Previous Frame", self)
    previous_action.setShortcut("Left")
    previous_action.triggered.connect(self.previous_frame)
    playback_menu.addAction(previous_action)

    next_action = QAction("Next Frame", self)
    next_action.setShortcut("Right")
    next_action.triggered.connect(self.next_frame)
    playback_menu.addAction(next_action)

    playback_menu.addSeparator()

    playback_slider_action = QAction("Frame Slider", self)
    playback_slider_action.setShortcut("Ctrl+F")
    playback_slider_action.triggered.connect(self.show_playback_slider)
    playback_menu.addAction(playback_slider_action)

    playback_menu.addSeparator()

    jump_to_action = QAction("Jump To Frame...", self)
    jump_to_action.setShortcut("Ctrl+J")
    jump_to_action.triggered.connect(self.show_jump_to_frame_dialog)
    playback_menu.addAction(jump_to_action)

    # Set up the filters menu
    filters_menu = menubar.addMenu("Filters")

    grayscale_action = QAction("Grayscale", self)
    grayscale_action.triggered.connect(self.apply_grayscale_filter)
    filters_menu.addAction(grayscale_action)

    invert_action = QAction("Invert", self)
    invert_action.triggered.connect(self.apply_invert_filter)
    filters_menu.addAction(invert_action)

    # Set up the histogram menu
    histogram_menu = menubar.addMenu("Histogram")

    frame_histogram_action = QAction("Frame Histogram", self)
    frame_histogram_action.triggered.connect(self.show_frame_histogram)
    histogram_menu.addAction(frame_histogram_action)

    video_histogram_action = QAction("Video Histogram", self)
    video_histogram_action.triggered.connect(self.show_video_histogram)
    histogram_menu.addAction(video_histogram_action)

    # Set up the file menu
    file_menu = menubar.addMenu("File")

    open_action = QAction("Open", self)
    open_action.setShortcut("Ctrl+O")
    open_action.triggered.connect(self.open_video)
    file_menu.addAction(open_action)

    close_action = QAction("Close", self)
    close_action.setShortcut("Ctrl+W")
    close_action.triggered.connect(self.close_video)
    file_menu.addAction(close_action)

    file_menu.addSeparator()

    save_frame_action = QAction("Save Frame As...", self)
    save_frame_action.setShortcut("Ctrl+Shift+S")
    save_frame_action.triggered.connect(self.show_save_frame_dialog)
    file_menu.addAction(save_frame_action)

    # Set up the toolbar
    toolbar = self.addToolBar("Controls")

    toolbar.addAction(play_action)
    toolbar.addAction(pause_action)
    toolbar.addAction(stop_action)
    toolbar.addAction(previous_action)
    toolbar.addAction(next_action)
    toolbar.addSeparator()
    toolbar.addAction(playback_slider_action)
    toolbar.addSeparator()
    toolbar.addAction(jump_to_action)
    toolbar.addSeparator()
    toolbar.addAction(grayscale_action)
    toolbar.addAction(invert_action)
    toolbar.addSeparator()
    toolbar.addAction(frame_histogram_action)
    toolbar.addAction(video_histogram_action)

    # Set up the status bar
    self.statusBar()

    # Set up the playback slider
    self.playback_slider = QSlider(Qt.Horizontal)
    self.playback_slider.setFocusPolicy(Qt.NoFocus)
    self.playback_slider.setTickPosition(QSlider.TicksBothSides)
    self.playback_slider.setTickInterval(1)
    self.playback_slider.valueChanged.connect(self.playback_slider_changed)
    toolbar.addWidget(self.playback_slider)

    # Set up the playback speed label and slider
    self.playback_speed_label = QLabel("Playback Speed:")
    toolbar.addWidget(self.playback_speed_label)

    self.playback_speed_slider = QSlider(Qt.Horizontal)
    self.playback_speed_slider.setFocusPolicy(Qt.NoFocus)
    self.playback_speed_slider.setRange(0, 100)
    self.playback_speed_slider.setValue(50)
    self.playback_speed_slider.setTickPosition(QSlider.TicksBothSides)
    self.playback_speed_slider.setTickInterval(10)
    self.playback_speed_slider.valueChanged.connect(self.playback_speed_changed)
    toolbar.addWidget(self.playback_speed_slider)

    # Set up the frame rate label and slider
    self.frame_rate_label = QLabel("Frame Rate:")
    toolbar.addWidget(self.frame_rate_label)

    self.frame_rate_slider = QSlider(Qt.Horizontal)
    self.frame_rate_slider.setFocusPolicy(Qt.NoFocus)
    self.frame_rate_slider.setRange(0, 100)
    self.frame_rate_slider.setValue(50)
    self.frame_rate_slider.setTickPosition(QSlider.TicksBothSides)
    self.frame_rate_slider.setTickInterval(10)
    self.frame_rate_slider.valueChanged.connect(self.frame_rate_changed)
    toolbar.addWidget(self.frame_rate_slider)

    # Set up the filter menu
    self.filter_menu = self.menuBar().addMenu("Filter")

    self.gray_scale_filter_action = QAction("Gray Scale", self)
    self.gray_scale_filter_action.triggered.connect(lambda: self.apply_filter(self.video_player_library.gray_scale_filter))
    self.filter_menu.addAction(self.gray_scale_filter_action)

    self.invert_filter_action = QAction("Invert", self)
    self.invert_filter_action.triggered.connect(lambda: self.apply_filter(self.video_player_library.invert_filter))
    self.filter_menu.addAction(self.invert_filter_action)

    self.flip_horizontally_filter_action = QAction("Flip Horizontally", self)
    self.flip_horizontally_filter_action.triggered.connect(lambda: self.apply_filter(self.video_player_library.flip_horizontally_filter))
    self.filter_menu.addAction(self.flip_horizontally_filter_action)

    self.flip_vertically_filter_action = QAction("Flip Vertically", self)
    self.flip_vertically_filter_action.triggered.connect(lambda: self.apply_filter(self.video_player_library.flip_vertically_filter))
    self.filter_menu.addAction(self.flip_vertically_filter_action)

    self.blur_filter_action = QAction("Blur", self)
    self.blur_filter_action.triggered.connect(lambda: self.apply_filter(self.video_player_library.blur_filter))
    self.filter_menu.addAction(self.blur_filter_action)

    self.sharpen_filter_action = QAction("Sharpen", self)
    self.sharpen_filter_action.triggered.connect(lambda: self.apply_filter(self.video_player_library.sharpen_filter))
    self.filter_menu.addAction(self.sharpen_filter_action)

    self.edge_detection_filter_action = QAction("Edge Detection", self)
    self.edge_detection_filter_action.triggered.connect(lambda: self.apply_filter(self.video_player_library.edge_detection_filter))
    self.filter_menu.addAction(self.edge_detection_filter_action)

    self.custom_filter_action = QAction("Custom Filter", self)
    self.custom_filter_action.triggered.connect(self.show_custom_filter_dialog)
    self.filter_menu.addAction(self.custom_filter_action)

    # Set up the histogram menu
    self.histogram_menu = self.menuBar().addMenu("Histogram")

    self.frame_histogram_action = QAction("Frame Histogram", self)
    self.video_histogram_action = QAction("Video Histogram", self)

    self.frame_histogram_action.triggered.connect(self.show_frame_histogram)
    self.video_histogram_action.triggered.connect(self.show_video_histogram)

    self.histogram_menu.addAction(self.frame_histogram_action)
    self.histogram_menu.addAction(self.video_histogram_action)

    # Set up the filter menu
    self.filter_menu = self.menuBar().addMenu("Filter")

    self.apply_filter_action = QAction("Apply Filter", self)
    self.apply_filter_action.triggered.connect(self.open_filter_dialog)

    self.filter_menu.addAction(self.apply_filter_action)

    # Set up the status bar
    self.statusBar()

    # Set up the timer
    self.timer = QTimer(self)
    self.timer.timeout.connect(self.update_frame)

    # Set up the video playback state
    self.playing = False
    self.paused = False

    # Set up the playback slider
    self.playback_slider = QSlider(Qt.Horizontal)
    self.playback_slider.setFocusPolicy(Qt.NoFocus)
    self.playback_slider.sliderMoved.connect(self.playback_slider_changed)
    self.playback_slider.setTickInterval(1)
    self.playback_slider.setSingleStep(1)

    toolbar.addWidget(self.playback_slider)

    # Set up the filter dialog
    self.filter_dialog = QDialog(self)
    self.filter_dialog.setWindowTitle("Apply Filter")
    self.filter_dialog.setFixedSize(400, 300)

    filter_label = QLabel("Select a filter to apply:")
    self.filter_list = QListWidget()
    self.filter_list.addItems(["Grayscale", "Canny", "Sobel X", "Sobel Y", "Laplacian", "Blur"])
    self.filter_button = QPushButton("Apply Filter")
    self.filter_button.clicked.connect(self.apply_selected_filter)

    filter_layout = QVBoxLayout()
    filter_layout.addWidget(filter_label)
    filter_layout.addWidget(self.filter_list)
    filter_layout.addWidget(self.filter_button)

    self.filter_dialog.setLayout(filter_layout)

    # Set up the video player library
    self.video_player_library = None

def update_ui(self):
    if self.video_player_library is None:
        self.setWindowTitle("Video Player")
        self.setGeometry(100, 100, 300, 50)
        self.play_button.setEnabled(False)
        self.pause_button.setEnabled(False)
        self.stop_button.setEnabled(False)
        self.previous_button.setEnabled(False)
        self.next_button.setEnabled(False)
        self.playback_slider.setEnabled(False)
        self.playback_slider.setMaximum(0)
        self.statusBar().showMessage("No video loaded.")
    else:
        self.setWindowTitle("Video Player - " + os.path.basename(self.video_player_library.get_video_path()))
        self.setGeometry(100, 100, 800, 600)
        self.play_button.setEnabled(True)
        self.pause_button.setEnabled(True)
        self.stop_button.setEnabled(True)
        self.previous_button.setEnabled(True)
        self.next_button.setEnabled(True)
        self.playback_slider.setEnabled(True)
        self.playback_slider.setMaximum(self.video_player_library.get_total_frames() - 1)
        self.statusBar().showMessage("Ready to play video.")

        # Update the video label
        self.update_frame()

def update_frame(self):
    if self.video_player_library is None:
        return

    # Get the current frame and convert it to a QImage
    frame = self.video_player_library.get_frame()
    if frame is not None:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, channel = frame.shape
        bytesPerLine = 3 * width
        qImg = QImage(frame.data, width, height, bytesPerLine, QImage.Format_RGB888)
        pixmap = QPixmap(qImg)

        # Update the video label with the new pixmap
        self.video_label.setPixmap(pixmap)

    # Update the playback slider position
    self.playback_slider.setValue(self.video_player_library.get_frame_index())

def play_video(self):
    if self.video_player_library is None:
        return

    if self.video_player_library.is_playing() or self.video_player_library.is_paused():
        self.video_player_library.play()
    else:
        self.video_player_library.start_playback()

    self.timer.start(0)

def pause_video(self):
    if self.video_player_library is None:
        return

    self.video_player_library.pause()

def stop_video(self):
    if self.video_player_library is None:
        return

    self.video_player_library.stop_playback()
    self.playback_slider.setValue(0)
    self.timer.stop()

def previous_frame(self):
    if self.video_player_library is None:
        return

    current_frame_index = self.video_player_library.get_frame_index()
    if current_frame_index > 0:
        self.video_player_library.seek_frame(current_frame_index - 1)
        self.update_frame()

def next_frame(self):
    if self.video_player_library is None:
        return

    current_frame_index = self.video_player_library.get_frame_index()
    total_frames = self.video_player_library.get_total_frames()
    if current_frame_index < total_frames - 1:
        self.video_player_library.seek_frame(current_frame_index + 1)
        self.update_frame()

def seek_frame(self):
    if self.video_player_library is None:
        return

    frame_index = self.playback_slider.value()
    self.video_player_library.seek_frame(frame_index)
    self.update_frame()

    def save_frame(self):
        if self.video_player_library is None:
            return

        file_path, _ = QFileDialog.getSaveFileName(self, "Save Frame As", "", "PNG Image (*.png)")
        if file_path:
            frame_index = self.video_player_library.get_frame_index()
            frame = self.video_player_library.get_frame()

            if frame is not None:
                cv2.imwrite(file_path, frame)
                self.statusBar().showMessage(f"Frame saved as {file_path}")

    def apply_filter(self, filter_func):
        if self.video_player_library is None:
            return

        self.video_player_library.apply_filter(filter_func)
        self.update_ui()

    def show_frame_histogram(self):
        if self.video_player_library is None:
            return

        frame_index = self.video_player_library.get_frame_index()
        histogram = self.video_player_library.get_frame_histogram(frame_index)

        if histogram is not None:
            plt.plot(histogram)
            plt.xlim([0, 256])
            plt.title("Frame Histogram")
            plt.show()

    def show_video_histogram(self):
        if self.video_player_library is None:
            return

        histogram = self.video_player_library.get_video_histogram()

        if histogram is not None:
            plt.plot(histogram)
            plt.xlim([0, 256])
            plt.title("Video Histogram")
            plt.show()

    def playback_slider_changed(self, value):
        if self.video_player_library is None:
            return

        self.video_player_library.seek_frame(value)
        self.update_ui()

    def update_frame(self):
        if self.video_player_library is None:
            return

        frame = self.video_player_library.get_frame()
        if frame is not None:
            height, width, channels = frame.shape
            bytes_per_line = channels * width
            q_image = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_image)
            self.video_label.setPixmap(pixmap)
            self.playback_slider.setValue(self.video_player_library.get_frame_index())
        else:
            self.stop_video()

    def open_video(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Video File", "", "Video Files (*.mp4 *.avi)")

        if file_path:
            self.video_player_library = VideoPlayerLibrary(file_path)
            self.playing = False
            self.paused = False
            self.timer.stop()
            self.update_ui()
            self.playback_slider.setRange(0, self.video_player_library.get_total_frames() - 1)
            self.playback_slider.setEnabled(True)
            self.statusBar().showMessage(f"Video loaded: {file_path}")
        else:
            self.statusBar().showMessage("No video selected")

    def close_video(self):
        self.video_player_library = None
        self.playing = False
        self.paused = False
        self.timer.stop()
        self.update_ui()
        self.playback_slider.setRange(0, 0)
        self.playback_slider.setEnabled(False)
        self.statusBar().showMessage("Video closed")

