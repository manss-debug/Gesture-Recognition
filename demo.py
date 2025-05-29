import argparse
import sys
import time
from collections import deque
from multiprocessing import Manager, Process, Value
from typing import Optional, Tuple

import onnxruntime as ort
from loguru import logger

ort.set_default_logger_severity(4)
logger.add(sys.stdout, format="{level} | {message}")
logger.remove(0)

import cv2
import numpy as np
from omegaconf import OmegaConf

from constants import classes

# ─── UI & Mediapipe imports ─────────────────────────────────────────────────────
import threading
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QTextEdit,
    QPushButton, QHBoxLayout, QVBoxLayout, QSizePolicy
)
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap, QMovie, QFont

import mediapipe as mp
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
# ────────────────────────────────────────────────────────────────────────────────


class BaseRecognition:
    def __init__(self, model_path: str, tensors_list, prediction_list, verbose):
        self.verbose = verbose
        self.started = None
        self.output_names = None
        self.input_shape = None
        self.input_name = None
        self.session = None
        self.model_path = model_path
        self.window_size = None
        self.tensors_list = tensors_list
        self.prediction_list = prediction_list

    def clear_tensors(self):
        # Очистка очереди кадров для ListProxy и обычного списка
        try:
            # для обычного списка или deque
            self.tensors_list.clear()
        except AttributeError:
            # для Manager.list() — удаляем все элементы
            del self.tensors_list[:]

    def run(self):
        if self.session is None:
            self.session = ort.InferenceSession(
                self.model_path,
                providers=["CPUExecutionProvider"]
            )
            self.input_name = self.session.get_inputs()[0].name
            self.input_shape = self.session.get_inputs()[0].shape
            self.window_size = self.input_shape[3]
            self.output_names = [o.name for o in self.session.get_outputs()]

        if len(self.tensors_list) >= self.window_size:
            input_tensor = np.stack(self.tensors_list[:self.window_size], axis=1)[None][None]
            st = time.time()
            outputs = self.session.run(
                self.output_names,
                {self.input_name: input_tensor.astype(np.float32)}
            )[0]
            et = round(time.time() - st, 3)
            gloss = str(classes[outputs.argmax()])
            if gloss != self.prediction_list[-1] and len(self.prediction_list):
                if gloss != "---":
                    self.prediction_list.append(gloss)
            self.clear_tensors()
            if self.verbose:
                logger.info(f"─ Prediction time {et}s, new gloss: {gloss}")
                logger.info(f"─ {len(self.tensors_list)} frames in queue")

    def kill(self):
        pass


class Recognition(BaseRecognition):
    def __init__(self, model_path: str, tensors_list: list, prediction_list: list, verbose: bool):
        super().__init__(
            model_path=model_path,
            tensors_list=tensors_list,
            prediction_list=prediction_list,
            verbose=verbose
        )
        self.started = True

    def start(self):
        self.run()


class RecognitionMP(Process, BaseRecognition):
    def __init__(self, model_path: str, tensors_list, prediction_list, verbose: bool):
        super().__init__()
        BaseRecognition.__init__(
            self,
            model_path=model_path,
            tensors_list=tensors_list,
            prediction_list=prediction_list,
            verbose=verbose
        )
        self.started = Value("i", False)

    def run(self):
        while True:
            BaseRecognition.run(self)
            self.started = True


class Runner:
    STACK_SIZE = 6

    def __init__(
        self,
        model_path: str,
        config: OmegaConf = None,
        mp: bool = False,
        verbose: bool = False,
        length: int = STACK_SIZE,
    ) -> None:
        self.multiprocess = mp
        self.cap = cv2.VideoCapture(0)
        self.manager = Manager() if self.multiprocess else None
        self.tensors_list = self.manager.list() if self.multiprocess else []
        self.prediction_list = self.manager.list() if self.multiprocess else []
        self.prediction_list.append("---")
        self.frame_counter = 0
        self.frame_interval = config.frame_interval
        self.length = length
        self.mean = np.array(config.mean)
        self.std = np.array(config.std)
        if self.multiprocess:
            self.recognizer = RecognitionMP(
                model_path, self.tensors_list, self.prediction_list, verbose
            )
        else:
            self.recognizer = Recognition(
                model_path, self.tensors_list, self.prediction_list, verbose
            )

    def add_frame(self, image):
        self.frame_counter += 1
        if self.frame_counter == self.frame_interval:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = self.resize(image, (224, 224))
            image = (image - self.mean) / self.std
            image = np.transpose(image, [2, 0, 1])
            self.tensors_list.append(image)
            self.frame_counter = 0

    @staticmethod
    def resize(im, new_shape=(224, 224)):
        shape = im.shape[:2]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
        dw /= 2; dh /= 2
        if shape[::-1] != new_unpad:
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - .1)), int(round(dh + .1))
        left, right = int(round(dw - .1)), int(round(dw + .1))
        im = cv2.copyMakeBorder(
            im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(18, 18, 18)
        )
        return im

    def run(self):
        if self.multiprocess:
            self.recognizer.start()
        while self.cap.isOpened():
            if self.recognizer.started:
                _, frame = self.cap.read()
                text_div = np.zeros((50, frame.shape[1], 3), dtype=np.uint8)
                self.add_frame(frame)
                if not self.multiprocess:
                    self.recognizer.start()
                if self.prediction_list:
                    text = "  ".join(self.prediction_list)
                    cv2.putText(
                        text_div, text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (230, 230, 230), 2
                    )
                if len(self.prediction_list) > self.length:
                    self.prediction_list.pop(0)
                frame = np.concatenate((frame, text_div), axis=0)
                cv2.imshow("frame", frame)
                if cv2.waitKey(10) & 0xFF in {ord("q"), ord("Q"), 27}:
                    if self.multiprocess:
                        self.recognizer.kill()
                    self.cap.release()
                    cv2.destroyAllWindows()
                    break


def parse_arguments(params: Optional[Tuple] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Demo full frame classification with neon UI...")
    parser.add_argument("-p", "--config", required=True, type=str, help="Path to config")
    parser.add_argument("--mp", action="store_true", help="Enable multiprocessing")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable logging")
    parser.add_argument("-l", "--length", type=int, default=4, help="Deque length for predictions")
    known_args, _ = parser.parse_known_args(params)
    return known_args


class GestureGUI(QWidget):
    def __init__(self, runner: Runner):
        super().__init__()
        self.runner = runner
        self.setWindowTitle("Gesture Recognition UI")
        self.setStyleSheet("background:#0f0f0f; color:#ddd;")
        self.setMinimumSize(1920, 1080)
        self.resize(1920, 1080)

        # Видео
        self.video_label = QLabel()
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.video_label.setStyleSheet(
            "border:3px solid #2d2d2d; border-radius:15px; background:#111;"
        )

        # Prediction label
        self.pred_label = QLabel("---")
        self.pred_label.setAlignment(Qt.AlignCenter)
        self.pred_label.setFont(QFont("Segoe UI", 36, QFont.Bold))
        self.pred_label.setStyleSheet(
            "background:linear-gradient(90deg, #00f5ff, #3a00ff); color:white;"
            "border-radius:20px; padding:15px;"
        )
        self.pred_label.setFixedHeight(80)

        # GIF + text input
        self.gif_label = QLabel()
        self.gif_label.setFixedSize(540, 540)
        self.gif_label.setStyleSheet(
            "border:3px solid #2d2d2d; border-radius:15px; background:#111;"
        )
        self.text_edit = QTextEdit()
        self.text_edit.setFixedHeight(100)
        self.text_edit.setPlaceholderText("Введите слово для GIF...")
        self.text_edit.setStyleSheet(
            "background:#222; color:#eee; border-radius:10px; padding:10px;"
        )
        self.send_btn = QPushButton("Загрузить GIF")
        self.send_btn.setFixedHeight(60)
        self.send_btn.setStyleSheet(
            "background:#3a00ff; color:white; font-size:18px;"
            "border-radius:15px;"
        )
        self.send_btn.clicked.connect(self.on_send)

        # Layouts
        right = QVBoxLayout()
        right.addWidget(self.gif_label, alignment=Qt.AlignTop)
        right.addWidget(self.text_edit)
        right.addWidget(self.send_btn, alignment=Qt.AlignRight)

        left = QVBoxLayout()
        left.addWidget(self.video_label)
        left.addWidget(self.pred_label)

        main = QHBoxLayout(self)
        main.addLayout(left, stretch=5)
        main.addLayout(right, stretch=3)
        self.setLayout(main)

        # Mediapipe hand tracker
        self.hands = mp_hands.Hands(
            max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.5
        )

        # Timer for frame updates
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(16)  # ~60fps

    def update_frame(self):
        ret, frame = self.runner.cap.read()
        if not ret:
            return

        # Draw hand skeleton
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = self.hands.process(rgb)
        if res.multi_hand_landmarks:
            for lm in res.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, lm, mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec((0,255,255),3,6),
                    mp_drawing.DrawingSpec((255,0,255),3,6)
                )

        # Feed frame and run recognition
        self.runner.add_frame(frame)
        if not self.runner.multiprocess:
            self.runner.recognizer.start()

        latest = self.runner.prediction_list[-1] if self.runner.prediction_list else "---"
        self.pred_label.setText(latest if latest != "---" else "-")

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = img.shape
        bytes_per = ch * w
        qimg = QImage(img.data, w, h, bytes_per, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qimg).scaled(
            self.video_label.width(), self.video_label.height(),
            Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.video_label.setPixmap(pix)

    def on_send(self):
        word = self.text_edit.toPlainText().strip().lower()
        if not word:
            self.gif_label.clear()
            return
        path = f"videos/{word}.gif"
        movie = QMovie(path)
        if movie.isValid():
            self.gif_label.setMovie(movie)
            movie.start()
        else:
            self.gif_label.clear()


if __name__ == "__main__":
    args = parse_arguments()
    conf = OmegaConf.load(args.config)

    runner = Runner(conf.model_path, conf, args.mp, args.verbose, args.length)
    if args.mp:
        runner.recognizer.start()

    app = QApplication(sys.argv)
    gui = GestureGUI(runner)
    gui.show()
    sys.exit(app.exec_())
