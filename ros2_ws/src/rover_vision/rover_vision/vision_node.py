#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from collections import deque
from collections import deque
import cv2
import math
import math
import numpy as np
import pytesseract

import threading


class VisionNode(Node):

    def __init__(self):
        super().__init__('vision_node')
        self.declare_parameter('detections_topic', '/detections')
        self.declare_parameter('camera_index', 0)
        self.declare_parameter('frame_width', 640)
        self.declare_parameter('frame_height', 480)
        self.declare_parameter('fps', 30)
        self.declare_parameter('timer_period_s', 0.1)
        self.declare_parameter('enable_gui', False)
        self.declare_parameter('enable_people_detection', True)
        self.declare_parameter('people_resize_width', 320)
        self.declare_parameter('people_resize_height', 240)
        self.declare_parameter('enable_ocr', True)
        self.declare_parameter('ocr_every_n_frames', 15)
        self.declare_parameter('enable_panel_detection', True)
        self.declare_parameter('rock_min_area_px', 700)
        self.declare_parameter('rock_morph_kernel_size', 5)
        self.declare_parameter('rock_morph_iterations', 1)
        self.declare_parameter('rock_min_solidity', 0.65)
        self.declare_parameter('rock_min_extent', 0.20)
        self.declare_parameter('rock_min_circularity', 0.05)
        self.declare_parameter('rock_enable_temporal_filter', True)
        self.declare_parameter('rock_persistence_frames', 3)
        self.declare_parameter('rock_smoothing_window', 5)
        self.declare_parameter('rock_publish_every_n_frames', 2)

        self._detections_topic = (
            self.get_parameter('detections_topic').get_parameter_value().string_value
        )
        self._camera_index = (
            self.get_parameter('camera_index').get_parameter_value().integer_value
        )
        self._frame_width = (
            self.get_parameter('frame_width').get_parameter_value().integer_value
        )
        self._frame_height = (
            self.get_parameter('frame_height').get_parameter_value().integer_value
        )
        self._fps = self.get_parameter('fps').get_parameter_value().integer_value
        self._timer_period_s = (
            self.get_parameter('timer_period_s').get_parameter_value().double_value
        )
        self._enable_gui = (
            self.get_parameter('enable_gui').get_parameter_value().bool_value
        )
        self._enable_people_detection = (
            self.get_parameter('enable_people_detection')
            .get_parameter_value()
            .bool_value
        )
        self._people_resize_width = (
            self.get_parameter('people_resize_width')
            .get_parameter_value()
            .integer_value
        )
        self._people_resize_height = (
            self.get_parameter('people_resize_height')
            .get_parameter_value()
            .integer_value
        )
        self._enable_ocr = (
            self.get_parameter('enable_ocr').get_parameter_value().bool_value
        )
        self._ocr_every_n_frames = (
            self.get_parameter('ocr_every_n_frames')
            .get_parameter_value()
            .integer_value
        )
        self._enable_panel_detection = (
            self.get_parameter('enable_panel_detection')
            .get_parameter_value()
            .bool_value
        )
        self._rock_min_area_px = (
            self.get_parameter('rock_min_area_px').get_parameter_value().integer_value
        )
        self._rock_morph_kernel_size = (
            self.get_parameter('rock_morph_kernel_size')
            .get_parameter_value()
            .integer_value
        )
        self._rock_morph_iterations = (
            self.get_parameter('rock_morph_iterations')
            .get_parameter_value()
            .integer_value
        )
        self._rock_min_solidity = (
            self.get_parameter('rock_min_solidity').get_parameter_value().double_value
        )
        self._rock_min_extent = (
            self.get_parameter('rock_min_extent').get_parameter_value().double_value
        )
        self._rock_min_circularity = (
            self.get_parameter('rock_min_circularity')
            .get_parameter_value()
            .double_value
        )
        self._rock_enable_temporal_filter = (
            self.get_parameter('rock_enable_temporal_filter')
            .get_parameter_value()
            .bool_value
        )
        self._rock_persistence_frames = (
            self.get_parameter('rock_persistence_frames')
            .get_parameter_value()
            .integer_value
        )
        self._rock_smoothing_window = (
            self.get_parameter('rock_smoothing_window')
            .get_parameter_value()
            .integer_value
        )
        self._rock_publish_every_n_frames = (
            self.get_parameter('rock_publish_every_n_frames')
            .get_parameter_value()
            .integer_value
        )

        self.publisher_ = self.create_publisher(String, self._detections_topic, 10)
        self.cap = cv2.VideoCapture(int(self._camera_index))

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(self._frame_width))
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(self._frame_height))
        self.cap.set(cv2.CAP_PROP_FPS, int(self._fps))
        self.declare_parameter('detections_topic', '/detections')
        self.declare_parameter('camera_index', 0)
        self.declare_parameter('frame_width', 640)
        self.declare_parameter('frame_height', 480)
        self.declare_parameter('fps', 30)
        self.declare_parameter('timer_period_s', 0.1)
        self.declare_parameter('enable_gui', False)
        self.declare_parameter('enable_people_detection', True)
        self.declare_parameter('people_resize_width', 320)
        self.declare_parameter('people_resize_height', 240)
        self.declare_parameter('enable_ocr', True)
        self.declare_parameter('ocr_every_n_frames', 15)
        self.declare_parameter('enable_panel_detection', True)
        self.declare_parameter('rock_min_area_px', 700)
        self.declare_parameter('rock_morph_kernel_size', 5)
        self.declare_parameter('rock_morph_iterations', 1)
        self.declare_parameter('rock_min_solidity', 0.65)
        self.declare_parameter('rock_min_extent', 0.20)
        self.declare_parameter('rock_min_circularity', 0.05)
        self.declare_parameter('rock_enable_temporal_filter', True)
        self.declare_parameter('rock_persistence_frames', 3)
        self.declare_parameter('rock_smoothing_window', 5)
        self.declare_parameter('rock_publish_every_n_frames', 2)

        self._detections_topic = (
            self.get_parameter('detections_topic').get_parameter_value().string_value
        )
        self._camera_index = (
            self.get_parameter('camera_index').get_parameter_value().integer_value
        )
        self._frame_width = (
            self.get_parameter('frame_width').get_parameter_value().integer_value
        )
        self._frame_height = (
            self.get_parameter('frame_height').get_parameter_value().integer_value
        )
        self._fps = self.get_parameter('fps').get_parameter_value().integer_value
        self._timer_period_s = (
            self.get_parameter('timer_period_s').get_parameter_value().double_value
        )
        self._enable_gui = (
            self.get_parameter('enable_gui').get_parameter_value().bool_value
        )
        self._enable_people_detection = (
            self.get_parameter('enable_people_detection')
            .get_parameter_value()
            .bool_value
        )
        self._people_resize_width = (
            self.get_parameter('people_resize_width')
            .get_parameter_value()
            .integer_value
        )
        self._people_resize_height = (
            self.get_parameter('people_resize_height')
            .get_parameter_value()
            .integer_value
        )
        self._enable_ocr = (
            self.get_parameter('enable_ocr').get_parameter_value().bool_value
        )
        self._ocr_every_n_frames = (
            self.get_parameter('ocr_every_n_frames')
            .get_parameter_value()
            .integer_value
        )
        self._enable_panel_detection = (
            self.get_parameter('enable_panel_detection')
            .get_parameter_value()
            .bool_value
        )
        self._rock_min_area_px = (
            self.get_parameter('rock_min_area_px').get_parameter_value().integer_value
        )
        self._rock_morph_kernel_size = (
            self.get_parameter('rock_morph_kernel_size')
            .get_parameter_value()
            .integer_value
        )
        self._rock_morph_iterations = (
            self.get_parameter('rock_morph_iterations')
            .get_parameter_value()
            .integer_value
        )
        self._rock_min_solidity = (
            self.get_parameter('rock_min_solidity').get_parameter_value().double_value
        )
        self._rock_min_extent = (
            self.get_parameter('rock_min_extent').get_parameter_value().double_value
        )
        self._rock_min_circularity = (
            self.get_parameter('rock_min_circularity')
            .get_parameter_value()
            .double_value
        )
        self._rock_enable_temporal_filter = (
            self.get_parameter('rock_enable_temporal_filter')
            .get_parameter_value()
            .bool_value
        )
        self._rock_persistence_frames = (
            self.get_parameter('rock_persistence_frames')
            .get_parameter_value()
            .integer_value
        )
        self._rock_smoothing_window = (
            self.get_parameter('rock_smoothing_window')
            .get_parameter_value()
            .integer_value
        )
        self._rock_publish_every_n_frames = (
            self.get_parameter('rock_publish_every_n_frames')
            .get_parameter_value()
            .integer_value
        )

        self.publisher_ = self.create_publisher(String, self._detections_topic, 10)
        self.cap = cv2.VideoCapture(int(self._camera_index))

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(self._frame_width))
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(self._frame_height))
        self.cap.set(cv2.CAP_PROP_FPS, int(self._fps))

        self.frame_count = 0
        self.ocr_running = False
        self._ocr_lock = threading.Lock()

        if not self.cap.isOpened():
            self.get_logger().error("No se pudo abrir la cámara")

        self.hog = None
        if self._enable_people_detection:
            self.hog = cv2.HOGDescriptor()
            self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

        smoothing_window = max(int(self._rock_smoothing_window), 1)
        self._rock_tracks = {
            'rojo': {'hits': 0, 'cxs': deque(maxlen=smoothing_window)},
            'azul': {'hits': 0, 'cxs': deque(maxlen=smoothing_window)},
            'verde': {'hits': 0, 'cxs': deque(maxlen=smoothing_window)},
        }

        self.timer = self.create_timer(self._timer_period_s, self.process_frame)
        self.hog = None
        if self._enable_people_detection:
            self.hog = cv2.HOGDescriptor()
            self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

        smoothing_window = max(int(self._rock_smoothing_window), 1)
        self._rock_tracks = {
            'rojo': {'hits': 0, 'cxs': deque(maxlen=smoothing_window)},
            'azul': {'hits': 0, 'cxs': deque(maxlen=smoothing_window)},
            'verde': {'hits': 0, 'cxs': deque(maxlen=smoothing_window)},
        }

        self.timer = self.create_timer(self._timer_period_s, self.process_frame)

    def run_ocr(self, img_thresh):
        try:
            texto_detectado = pytesseract.image_to_string(img_thresh, config='--psm 11').strip().upper()
            if "FIN" in texto_detectado:
                msg_fin = String()
                msg_fin.data = "fin,320,negro,N/A" 
                self.publisher_.publish(msg_fin)
        except Exception:
            pass 
        finally:
            with self._ocr_lock:
                self.ocr_running = False

    def process_frame(self):
        if not self.cap.isOpened():
            return

        if not self.cap.isOpened():
            return

        ret, frame = self.cap.read()
        if not ret:
            return

        self.frame_count += 1

        if self._enable_people_detection and self.hog is not None:
            resize_w = max(int(self._people_resize_width), 1)
            resize_h = max(int(self._people_resize_height), 1)
            small_frame = cv2.resize(frame, (resize_w, resize_h))
            boxes, _weights = self.hog.detectMultiScale(
                small_frame,
                winStride=(8, 8),
                padding=(4, 4),
                scale=1.05,
            )
            if len(boxes) > 0:
                self.publisher_.publish(String(data="persona,stop_all"))
                cv2.putText(
                    frame,
                    "PERSONA DETECTADA - STOP ALL",
                    (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    3,
                )

        if self._enable_ocr and self._ocr_every_n_frames > 0:
            if self.frame_count % int(self._ocr_every_n_frames) == 0:
                roi_h = min(240, frame.shape[0])
                roi_w = min(640, frame.shape[1])
                roi_top = frame[0:roi_h, 0:roi_w]
                gray = cv2.cvtColor(roi_top, cv2.COLOR_BGR2GRAY)
                _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
                try:
                    texto_detectado = (
                        pytesseract.image_to_string(thresh, config='--psm 11')
                        .strip()
                        .upper()
                    )
                    if "FIN" in texto_detectado:
                        self.publisher_.publish(String(data="fin,320,negro,N/A"))
                        cv2.putText(
                            frame,
                            "LETRERO FIN ENCONTRADO",
                            (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 0, 255),
                            3,
                        )
                except Exception as exc:
                    self.get_logger().warning(f'Error OCR: {exc}')

<<<<<<< HEAD
        # =======================================================
        # OCR (Letrero FIN) - Se ejecuta asíncronamente
        # =======================================================
        start_ocr = False
        if self.frame_count % 15 == 0:
            with self._ocr_lock:
                if not self.ocr_running:
                    self.ocr_running = True
                    start_ocr = True

        if start_ocr:
            roi_top = frame[0:240, 0:640] 
            gray = cv2.cvtColor(roi_top, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
            
            threading.Thread(target=self.run_ocr, args=(thresh,), daemon=True).start()
=======
        if self._enable_people_detection and self.hog is not None:
            resize_w = max(int(self._people_resize_width), 1)
            resize_h = max(int(self._people_resize_height), 1)
            small_frame = cv2.resize(frame, (resize_w, resize_h))
            boxes, _weights = self.hog.detectMultiScale(
                small_frame,
                winStride=(8, 8),
                padding=(4, 4),
                scale=1.05,
            )
            if len(boxes) > 0:
                self.publisher_.publish(String(data="persona,stop_all"))
                cv2.putText(
                    frame,
                    "PERSONA DETECTADA - STOP ALL",
                    (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    3,
                )

        if self._enable_ocr and self._ocr_every_n_frames > 0:
            if self.frame_count % int(self._ocr_every_n_frames) == 0:
                roi_h = min(240, frame.shape[0])
                roi_w = min(640, frame.shape[1])
                roi_top = frame[0:roi_h, 0:roi_w]
                gray = cv2.cvtColor(roi_top, cv2.COLOR_BGR2GRAY)
                _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
                try:
                    texto_detectado = (
                        pytesseract.image_to_string(thresh, config='--psm 11')
                        .strip()
                        .upper()
                    )
                    if "FIN" in texto_detectado:
                        self.publisher_.publish(String(data="fin,320,negro,N/A"))
                        cv2.putText(
                            frame,
                            "LETRERO FIN ENCONTRADO",
                            (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 0, 255),
                            3,
                        )
                except Exception as exc:
                    self.get_logger().warning(f'Error OCR: {exc}')
>>>>>>> 484671a941b23dc43508dfd8c199dd01c40009cb

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        mask_red1 = cv2.inRange(
            hsv, np.array([0, 100, 100]), np.array([10, 255, 255])
        )
        mask_red2 = cv2.inRange(
            hsv, np.array([160, 100, 100]), np.array([180, 255, 255])
        )
        mask_red1 = cv2.inRange(
            hsv, np.array([0, 100, 100]), np.array([10, 255, 255])
        )
        mask_red2 = cv2.inRange(
            hsv, np.array([160, 100, 100]), np.array([180, 255, 255])
        )
        mask_red = mask_red1 + mask_red2

        mask_blue = cv2.inRange(
            hsv, np.array([100, 150, 50]), np.array([140, 255, 255])
        )
        mask_green = cv2.inRange(
            hsv, np.array([40, 70, 70]), np.array([80, 255, 255])
        )
        mask_blue = cv2.inRange(
            hsv, np.array([100, 150, 50]), np.array([140, 255, 255])
        )
        mask_green = cv2.inRange(
            hsv, np.array([40, 70, 70]), np.array([80, 255, 255])
        )

        self.detect_color(mask_red, frame, "rojo")
        self.detect_color(mask_blue, frame, "azul")
        self.detect_color(mask_green, frame, "verde")

        if self._enable_panel_detection:
            mask_gray = cv2.inRange(
                hsv, np.array([0, 0, 50]), np.array([180, 50, 200])
            )
            contours_gray, _ = cv2.findContours(
                mask_gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
            )

            if contours_gray:
                largest_gray = max(contours_gray, key=cv2.contourArea)
                if cv2.contourArea(largest_gray) > 5000:
                    x, y, w, h = cv2.boundingRect(largest_gray)
                    cx = int(x + w / 2)
                    altura_estimada_mm = int(1000 - (y * 2))
                    self.publisher_.publish(
                        String(data=f"panel,{cx},{altura_estimada_mm}")
                    )

                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 255), 2)
                    cv2.putText(
                        frame,
                        f"PANEL: {altura_estimada_mm}mm",
                        (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 0, 255),
                        2,
                    )
<<<<<<< HEAD
        # =======================================================
        # DETECCIÓN DEL PANEL DE MANTENIMIENTO
        # =======================================================
        mask_gray = cv2.inRange(hsv, np.array([0, 0, 50]), np.array([180, 50, 200]))
        contours_gray, _ = cv2.findContours(mask_gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours_gray:
            largest_gray = max(contours_gray, key=cv2.contourArea)
            if cv2.contourArea(largest_gray) > 5000:
                x, y, w, h = cv2.boundingRect(largest_gray)
                cx = int(x + w/2)
                cy = int(y + h/2)
                
                altura_estimada_mm = int(1000 - (y * 2)) 
                
                msg_panel = String()
                msg_panel.data = f"panel,{cx},{altura_estimada_mm}"
                self.publisher_.publish(msg_panel)
                
                cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,255), 2)
                cv2.putText(frame, f"PANEL: {altura_estimada_mm}mm", (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,255), 2)

        if self._enable_gui:
            try:
                cv2.imshow("Vision - Fat Rat", frame)
                cv2.waitKey(1)
            except Exception:
                self._enable_gui = False

    def _preprocess_rock_mask(self, mask):
        kernel_size = int(self._rock_morph_kernel_size)
        if kernel_size <= 1:
            return mask

        if kernel_size % 2 == 0:
            kernel_size += 1

        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
        )
        iterations = max(int(self._rock_morph_iterations), 1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=iterations)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=iterations)
        return mask

    def _select_best_rock_contour(self, contours):
        min_area = max(int(self._rock_min_area_px), 1)
        min_solidity = float(self._rock_min_solidity)
        min_extent = float(self._rock_min_extent)
        min_circularity = float(self._rock_min_circularity)

        best = None
        best_score = 0.0

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            rect_area = float(w * h)
            if rect_area <= 0.0:
                continue

            extent = float(area) / rect_area
            if extent < min_extent:
                continue

            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            if hull_area <= 0.0:
                continue

            solidity = float(area) / float(hull_area)
            if solidity < min_solidity:
                continue

            perimeter = cv2.arcLength(contour, True)
            if perimeter <= 0.0:
                continue

            circularity = (4.0 * math.pi * float(area)) / float(perimeter * perimeter)
            if circularity < min_circularity:
                continue

            score = float(area) * solidity
            if score > best_score:
                best_score = score
                best = (contour, x, y, w, h, area)

        return best

    def _update_rock_track(self, color_name, cx, detected):
        track = self._rock_tracks.get(color_name)
        if track is None:
            return None

        if detected:
            track['hits'] = min(
                int(track['hits']) + 1, max(int(self._rock_persistence_frames), 1)
            )
            track['cxs'].append(int(cx))
        else:
            track['hits'] = max(int(track['hits']) - 1, 0)
            if track['hits'] == 0:
                track['cxs'].clear()

        if not self._rock_enable_temporal_filter:
            return int(cx) if detected else None

        if int(track['hits']) < max(int(self._rock_persistence_frames), 1):
            return None

        if not track['cxs']:
            return None

        return int(round(float(np.median(list(track['cxs'])))))
                # =======================================================
                # DETECCIÓN DE CONTROLES (Dentro del área del Panel)
                # =======================================================
                panel_roi = frame[y:y+h, x:x+w]
                
                if panel_roi.size > 0:
                    gray_panel = cv2.cvtColor(panel_roi, cv2.COLOR_BGR2GRAY)
                    blur = cv2.GaussianBlur(gray_panel, (5,5), 0)
              
                    # --- DETECCIÓN DE BOTONES ---
                    circles = cv2.HoughCircles(
                        blur,
                        cv2.HOUGH_GRADIENT,
                        dp=1.2,
                        minDist=40,
                        param1=50,
                        param2=30,
                        minRadius=10,
                        maxRadius=50
                    )

                    if circles is not None:
                        circles = np.uint16(np.around(circles))
                        for c in circles[0, :2]:
                            cx_btn = int(x + c[0])
                            cy_btn = int(y + c[1])
                            r = c[2]
                            
                            roi_btn = gray_panel[max(c[1]-r,0):c[1]+r, max(c[0]-r,0):c[0]+r]
                            estado = "on" if np.mean(roi_btn) > 150 else "off"

                            msg = String()
                            msg.data = f"control,boton,{cx_btn},{estado}"
                            self.publisher_.publish(msg)

                            cv2.circle(frame, (cx_btn, cy_btn), r, (0,255,255), 2)
                            cv2.putText(frame, f"BTN {estado}", (cx_btn, cy_btn), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 2)

                    # --- DETECCIÓN DE INTERRUPTORES ---
                    edges = cv2.Canny(blur, 50, 150)
                    contours_ctrl, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    switches_detected = 0

                    for cnt in contours_ctrl:
                        area = cv2.contourArea(cnt)
                        if area < 300:
                            continue

                        x2, y2, w2, h2 = cv2.boundingRect(cnt)
                        aspect_ratio = h2 / float(w2) if w2 > 0 else 0

                        if aspect_ratio > 1.5 or aspect_ratio < 0.5:
                            cx_sw = int(x + x2 + w2/2)
                            estado = "on" if h2 > w2 else "off"

                            msg = String()
                            msg.data = f"control,interruptor,{cx_sw},{estado}"
                            self.publisher_.publish(msg)

                            cv2.rectangle(frame, (x + x2, y + y2), (x + x2 + w2, y + y2 + h2), (255,255,0), 2)
                            cv2.putText(frame, f"SW {estado}", (x + x2, y + y2 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 2)
                            switches_detected += 1

                        if switches_detected >= 2:
                            break

        # =======================================================
        # DETECCIÓN DE CONTENEDOR BLANCO (INICIO)
        # =======================================================
        mask_white = cv2.inRange(hsv, np.array([0, 0, 200]), np.array([180, 50, 255]))
        contours_white, _ = cv2.findContours(mask_white, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours_white:
            largest_white = max(contours_white, key=cv2.contourArea)
            if cv2.contourArea(largest_white) > 3000:
                x_w, y_w, w_w, h_w = cv2.boundingRect(largest_white)
                cx_w = int(x_w + w_w/2)
                
                msg_inicio = String()
                msg_inicio.data = f"inicio,{cx_w}"
                self.publisher_.publish(msg_inicio)
                
                cv2.rectangle(frame, (x_w,y_w), (x_w+w_w,y_w+h_w), (255,255,255), 2)
                cv2.putText(frame, "INICIO", (x_w,y_w-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)

        # Mostrar imagen
        try:
            cv2.imshow("Vision - Fat Rat", frame)
            cv2.waitKey(1)
        except Exception:
            pass
=======
        if self._enable_panel_detection:
            mask_gray = cv2.inRange(
                hsv, np.array([0, 0, 50]), np.array([180, 50, 200])
            )
            contours_gray, _ = cv2.findContours(
                mask_gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
            )

            if contours_gray:
                largest_gray = max(contours_gray, key=cv2.contourArea)
                if cv2.contourArea(largest_gray) > 5000:
                    x, y, w, h = cv2.boundingRect(largest_gray)
                    cx = int(x + w / 2)
                    altura_estimada_mm = int(1000 - (y * 2))
                    self.publisher_.publish(
                        String(data=f"panel,{cx},{altura_estimada_mm}")
                    )

                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 255), 2)
                    cv2.putText(
                        frame,
                        f"PANEL: {altura_estimada_mm}mm",
                        (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 0, 255),
                        2,
                    )

        if self._enable_gui:
            try:
                cv2.imshow("Vision - Fat Rat", frame)
                cv2.waitKey(1)
            except Exception:
                self._enable_gui = False

    def _preprocess_rock_mask(self, mask):
        kernel_size = int(self._rock_morph_kernel_size)
        if kernel_size <= 1:
            return mask

        if kernel_size % 2 == 0:
            kernel_size += 1

        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
        )
        iterations = max(int(self._rock_morph_iterations), 1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=iterations)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=iterations)
        return mask

    def _select_best_rock_contour(self, contours):
        min_area = max(int(self._rock_min_area_px), 1)
        min_solidity = float(self._rock_min_solidity)
        min_extent = float(self._rock_min_extent)
        min_circularity = float(self._rock_min_circularity)

        best = None
        best_score = 0.0

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            rect_area = float(w * h)
            if rect_area <= 0.0:
                continue

            extent = float(area) / rect_area
            if extent < min_extent:
                continue

            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            if hull_area <= 0.0:
                continue

            solidity = float(area) / float(hull_area)
            if solidity < min_solidity:
                continue

            perimeter = cv2.arcLength(contour, True)
            if perimeter <= 0.0:
                continue

            circularity = (4.0 * math.pi * float(area)) / float(perimeter * perimeter)
            if circularity < min_circularity:
                continue

            score = float(area) * solidity
            if score > best_score:
                best_score = score
                best = (contour, x, y, w, h, area)

        return best

    def _update_rock_track(self, color_name, cx, detected):
        track = self._rock_tracks.get(color_name)
        if track is None:
            return None

        if detected:
            track['hits'] = min(
                int(track['hits']) + 1, max(int(self._rock_persistence_frames), 1)
            )
            track['cxs'].append(int(cx))
        else:
            track['hits'] = max(int(track['hits']) - 1, 0)
            if track['hits'] == 0:
                track['cxs'].clear()

        if not self._rock_enable_temporal_filter:
            return int(cx) if detected else None

        if int(track['hits']) < max(int(self._rock_persistence_frames), 1):
            return None

        if not track['cxs']:
            return None

        return int(round(float(np.median(list(track['cxs'])))))
>>>>>>> 484671a941b23dc43508dfd8c199dd01c40009cb

    def detect_color(self, mask, frame, color_name):
        processed_mask = self._preprocess_rock_mask(mask)
        contours, _ = cv2.findContours(
            processed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        best = self._select_best_rock_contour(contours)
        if best is None:
            self._update_rock_track(color_name, 0, detected=False)
        processed_mask = self._preprocess_rock_mask(mask)
        contours, _ = cv2.findContours(
            processed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        best = self._select_best_rock_contour(contours)
        if best is None:
            self._update_rock_track(color_name, 0, detected=False)
            return

        contour, x, y, w, h, area = best
        cx = int(x + w / 2)
        filtered_cx = self._update_rock_track(color_name, cx, detected=True)
        if filtered_cx is None:
            if self._enable_gui:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
            return

        publish_every = max(int(self._rock_publish_every_n_frames), 1)
        if self.frame_count % publish_every != 0:
            if self._enable_gui:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
            return

        if area < 1500:
            tamano = "5cm3"
        elif area < 3000:
            tamano = "7cm3"
        elif area < 5000:
            tamano = "10cm3"
        else:
            tamano = "12cm3"

        self.publisher_.publish(String(data=f"roca,{filtered_cx},{color_name},{tamano}"))

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"{color_name} {tamano}",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )

    def destroy_node(self):
        try:
            if self.cap is not None:
                self.cap.release()
        except Exception:
            pass

        if self._enable_gui:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
        super().destroy_node()
        contour, x, y, w, h, area = best
        cx = int(x + w / 2)
        filtered_cx = self._update_rock_track(color_name, cx, detected=True)
        if filtered_cx is None:
            if self._enable_gui:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
            return

<<<<<<< HEAD
        if area > 500:
            x, y, w, h = cv2.boundingRect(largest_contour)
            cx = int(x + w/2)
            cy = int(y + h/2)

            # ============================
            # TAMAÑO
            # ============================
            if area < 1500: tamano = "5cm3"
            elif area < 3000: tamano = "7cm3"
            elif area < 5000: tamano = "10cm3"
            else: tamano = "12cm3"

            # ============================
            # TEXTURA
            # ============================
            roi_color = frame[y:y+h, x:x+w]

            if roi_color.size > 0:
                roi_gray = cv2.cvtColor(roi_color, cv2.COLOR_BGR2GRAY)
                varianza = cv2.Laplacian(roi_gray, cv2.CV_64F).var()
                textura = "rugosa" if varianza > 300 else "lisa"
            else:
                textura = "no_determinada"

            # ============================
            # FORMA
            # ============================
            peri = cv2.arcLength(largest_contour, True)
            approx = cv2.approxPolyDP(largest_contour, 0.04 * peri, True)

            num_vertices = len(approx)
            aspect_ratio = w / float(h) if h > 0 else 0

            if num_vertices >= 8:
                forma = "esferica"
            elif 4 <= num_vertices <= 6:
                forma = "cubica"
            elif aspect_ratio > 1.5 or aspect_ratio < 0.67:
                forma = "alargada"
            else:
                forma = "irregular"

            # ============================
            # PUBLICACIÓN
            # ============================
            msg = String()
            msg.data = f"roca,{cx},{cy},{color_name},{tamano},{textura},{forma}"
            self.publisher_.publish(msg)

            # Visualización
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
            cv2.putText(frame, f"{color_name} {forma}", (x,y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

    def destroy_node(self):
        try:
            if self.cap:
                self.cap.release()
        except Exception:
            pass
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
=======
        publish_every = max(int(self._rock_publish_every_n_frames), 1)
        if self.frame_count % publish_every != 0:
            if self._enable_gui:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
            return

        if area < 1500:
            tamano = "5cm3"
        elif area < 3000:
            tamano = "7cm3"
        elif area < 5000:
            tamano = "10cm3"
        else:
            tamano = "12cm3"

        self.publisher_.publish(String(data=f"roca,{filtered_cx},{color_name},{tamano}"))

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"{color_name} {tamano}",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )

    def destroy_node(self):
        try:
            if self.cap is not None:
                self.cap.release()
        except Exception:
            pass

        if self._enable_gui:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
>>>>>>> 484671a941b23dc43508dfd8c199dd01c40009cb
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = VisionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
