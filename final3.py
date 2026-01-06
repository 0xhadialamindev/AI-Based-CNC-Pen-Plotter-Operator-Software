import cv2
import numpy as np
import time
import serial
import serial.tools.list_ports
import re
from typing import List, Tuple, Optional
import threading
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import logging

# Setup logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Project Constants
PROJECT_NAME = "AI-Based CNC Operator for Portrait Images"
DESCRIPTION = """This project develops an end-to-end AI-powered system that captures portrait images via webcam, processes them using computer vision techniques (including AI-based background removal with OpenCV's GrabCut algorithm for semantic-like segmentation and edge detection with OpenCV), generates optimized sketches, converts them to G-code for CNC plotting, and executes the plotting on a mini CNC pen plotter. The AI focus lies in intelligent image capture (selecting sharpest frames using Laplacian variance), semantic-like segmentation for background removal, adaptive edge detection for sketch generation (tuned for minimalism and CNC efficiency with parameter variants), and real-time motor verification during plotting to handle hardware quirks like voltage sag. This ensures reliable, high-quality professor portraits on a small canvas, with user-friendly selection of processing variants for customization."""
CREDITS = "Built by Magura Polytechnic Institute, Session 2021-22, Mechatronics Team Lead: Md. Hadi Al-Amin (Roll 632792), under Dept. Head Samir Kundu Sir."
COPYRIGHT = "Copyright © 2026 by Md. Hadi Al-Amin"
THANKS = "Thanks to Samir Sir"

CAMERA_PORT = 0
CANVAS_WIDTH = 40.0
CANVAS_HEIGHT = 50.0
FEED_RATE = 50
BAUD = 115200
TOLERANCE = 0.3
COM_PORT = "COM4"  # Configurable
DRAW_SIZE = 32.0

# SmartCapture Class (modified without mediapipe)
class SmartCapture:
    def __init__(self, camera_id=None):
        self.cap = None
        if camera_id is not None:
            self.cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            # Warm up camera
            for _ in range(10):
                self.cap.read()
            logging.info("Camera warmed up")

    def capture_frames(self, num_frames=3, delay=2):
        if not self.cap:
            raise ValueError("No camera initialized")
        frames = []
        sharpness_scores = []
        for i in range(num_frames):
            time.sleep(delay if i > 0 else 0)
            ret, frame = self.cap.read()
            if not ret:
                logging.warning("Failed to capture frame")
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
            frames.append(frame)
            sharpness_scores.append(sharpness)
            logging.debug(f"Frame {i+1} sharpness: {sharpness}")
        return frames, sharpness_scores

    def sharpen_frame(self, frame):
        gaussian = cv2.GaussianBlur(frame, (0, 0), 3.0)
        return cv2.addWeighted(frame, 1.5, gaussian, -0.5, 0)

    def remove_background(self, frame, iterations=5):
        logging.info(f"Removing background with {iterations} iterations")
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        if len(faces) == 0:
            logging.warning("No face detected, returning original")
            return frame, np.zeros_like(gray)
        faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
        x, y, w, h = faces[0]
        margin = int(min(w, h) * 0.2)
        x = max(0, x - margin)
        y = max(0, y - margin)
        w += 2 * margin
        h += 2 * margin
        mask = np.zeros(frame.shape[:2], np.uint8)
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)
        rect = (x, y, w, h)
        cv2.grabCut(frame, mask, rect, bgdModel, fgdModel, iterations, cv2.GC_INIT_WITH_RECT)
        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
        person = frame * mask2[:, :, np.newaxis]
        white_bg = np.ones_like(frame) * 255
        white_bg = white_bg * (1 - mask2[:, :, np.newaxis])
        return cv2.add(person, white_bg), mask2 * 255

    def release(self):
        if self.cap:
            self.cap.release()
            logging.info("Camera released")

# RobustSketchGenerator with increased max_gap for better line connection
class RobustSketchGenerator:
    def __init__(self, canny_low=100, canny_high=250, min_area=50, dilate_iterations=1):
        self.canny_low = canny_low
        self.canny_high = canny_high
        self.min_area = min_area
        self.dilate_iterations = dilate_iterations
        self.hough_threshold = 20
        self.hough_min_line_length = 20
        self.hough_max_gap = 20  # Increased for better connection

    def generate_sketch(self, image):
        logging.info("Generating sketch")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        if len(faces) > 0:
            faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
            x, y, w, h = faces[0]
            margin = int(min(w, h) * 0.2)
            x = max(0, x - margin)
            y = max(0, y - margin)
            w += 2 * margin
            h += 2 * margin
            gray = gray[y:y+h, x:x+w]
            logging.debug(f"Cropped to face: {gray.shape}")
        gray = cv2.equalizeHist(gray)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)
        gamma = 1.2
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        gray = cv2.LUT(gray, table)
        h, w = gray.shape
        max_dim = 512
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            logging.debug(f"Resized to: {gray.shape}")
        blurred = cv2.bilateralFilter(gray, d=5, sigmaColor=50, sigmaSpace=50)
        edges = cv2.Canny(blurred, self.canny_low, self.canny_high)
        kernel = np.ones((3,3), np.uint8)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(edges, connectivity=8)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] < self.min_area:
                edges[labels == i] = 0
        lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=self.hough_threshold,
                                minLineLength=self.hough_min_line_length, maxLineGap=self.hough_max_gap)
        num_lines = len(lines) if lines is not None else 0
        logging.debug(f"Detected {num_lines} lines")
        if self.dilate_iterations > 0:
            dilate_kernel = np.ones((3,3), np.uint8)
            edges = cv2.dilate(edges, dilate_kernel, iterations=self.dilate_iterations)
        sketch = cv2.bitwise_not(edges)
        if np.mean(sketch) < 127:
            sketch = cv2.bitwise_not(sketch)
        return sketch, lines, num_lines

# Optimized G-Code Generation with line sorting for minimal travel
def sort_lines(lines):
    if lines is None or len(lines) == 0:
        return []
    # Treat each line as [start, end], allow reversing
    sorted_lines = [lines[0][0].tolist()]
    remaining = set(range(1, len(lines)))
    while remaining:
        last_end = sorted_lines[-1][2:4]
        closest_idx = None
        closest_dist = float('inf')
        closest_reverse = False
        for idx in remaining:
            line = lines[idx][0]
            dist_start = np.linalg.norm(np.array(last_end) - line[:2])
            dist_end = np.linalg.norm(np.array(last_end) - line[2:4])
            if dist_start < closest_dist:
                closest_dist = dist_start
                closest_idx = idx
                closest_reverse = False
            if dist_end < closest_dist:
                closest_dist = dist_end
                closest_idx = idx
                closest_reverse = True
        if closest_idx is None:
            break
        line = lines[closest_idx][0].tolist()
        if closest_reverse:
            line = line[2:4] + line[:2]
        sorted_lines.append(line)
        remaining.remove(closest_idx)
    logging.info(f"Optimized {len(lines)} lines to minimize travel")
    return sorted_lines

def generate_gcode_from_sketch(sketch, lines, size=DRAW_SIZE):
    logging.info("Generating G-code")
    if lines is None:
        return [], 0.0, 0.0
    optimized_lines = sort_lines(lines)
    height, width = sketch.shape
    scale_x = size / width
    scale_y = size / height
    start_x = (CANVAS_WIDTH - size) / 2
    start_y = (CANVAS_HEIGHT - size) / 2
    commands = [
        ("G21", None),
        ("G90", None),
        ("G92 X0 Y0", None),
    ]
    total_distance = 0.0
    pen_down = False
    current_pos = None
    for line in optimized_lines:
        x1, y1, x2, y2 = line
        gx1 = start_x + x1 * scale_x
        gy1 = start_y + (height - y1) * scale_y  # Invert Y
        gx2 = start_x + x2 * scale_x
        gy2 = start_y + (height - y2) * scale_y
        dist = np.sqrt((gx2 - gx1)**2 + (gy2 - gy1)**2)
        total_distance += dist
        if not pen_down or (current_pos and np.linalg.norm(np.array(current_pos) - np.array([gx1, gy1])) > 0.1):
            if pen_down:
                commands.append(("M3 S20", None))  # Pen up if was down
            commands.append((f"G0 X{gx1:.3f} Y{gy1:.3f}", (gx1, gy1)))
            commands.append(("M3 S0", None))  # Pen down
            pen_down = True
        commands.append((f"G1 X{gx2:.3f} Y{gy2:.3f}", (gx2, gy2)))
        current_pos = [gx2, gy2]
    if pen_down:
        commands.append(("M3 S20", None))
    commands.append(("G0 X0 Y0", (0, 0)))
    commands.append(("M18", None))
    estimated_time = total_distance / (FEED_RATE / 60.0) + len(optimized_lines) * 0.1 + len(commands) * 0.05
    logging.debug(f"Generated {len(commands)} commands, est time: {estimated_time:.1f}s")
    return commands, estimated_time, total_distance

# CNC Runner with pause support
class CNCRunner:
    def __init__(self, com_port=COM_PORT, feed_rate=FEED_RATE, baud=BAUD, tolerance=TOLERANCE):
        self.com_port = com_port
        self.feed_rate = feed_rate
        self.baud = baud
        self.tolerance = tolerance
        self.move_times = []
        self.paused = False
        self.cancelled = False

    def port(self):
        if self.com_port:
            return self.com_port
        for p in serial.tools.list_ports.comports():
            if any(kw in p.description.lower() for kw in ['arduino', 'ch340']):
                return p.device
        raise ValueError("No port found")

    def wait_ok(self, ser, timeout=5):
        start = time.time()
        while time.time() - start < timeout:
            if ser.in_waiting:
                if 'ok' in ser.readline().decode().strip().lower():
                    return True
            if self.cancelled:
                return False
        return False

    def get_pos(self, ser):
        ser.reset_input_buffer()
        ser.write(b'?\n')
        time.sleep(0.05)
        if ser.in_waiting:
            s = ser.readline().decode().strip()
            m = re.search(r'WPos:([\d\.\-]+),([\d\.\-]+)', s)
            if m:
                return float(m.group(1)), float(m.group(2))
        return None, None

    def wait_pos(self, ser, target, max_wait=20):
        if not target:
            return True
        start = time.time()
        while time.time() - start < max_wait:
            if self.cancelled:
                return False
            cur_x, cur_y = self.get_pos(ser)
            if cur_x is not None and cur_y is not None:
                if abs(cur_x - target[0]) < self.tolerance and abs(cur_y - target[1]) < self.tolerance:
                    return True
            time.sleep(0.05)
        return False

    def run(self, commands, progress_callback, total_distance, start_from=0):
        logging.info(f"Starting CNC run from command {start_from}")
        port = self.port()
        try:
            with serial.Serial(port, self.baud, timeout=2) as ser:
                time.sleep(2)
                ser.flushInput()
                ser.write(b"\r\n")
                current_pos = (0.0, 0.0)
                traveled = 0.0
                for i in range(start_from, len(commands)):
                    if self.cancelled:
                        logging.warning("CNC run cancelled")
                        break
                    while self.paused:
                        time.sleep(0.1)
                    cmd, target = commands[i]
                    logging.debug(f"Sending: {cmd}")
                    ser.write((cmd + '\n').encode())
                    if not self.wait_ok(ser, timeout=10 if cmd.startswith('G1') else 5):
                        logging.error("No OK received")
                        continue
                    if target:
                        if self.wait_pos(ser, target):
                            if current_pos != (0.0, 0.0):
                                dist = np.sqrt((target[0] - current_pos[0])**2 + (target[1] - current_pos[1])**2)
                                traveled += dist if cmd.startswith('G1') else 0
                            current_pos = target
                    if cmd.startswith('M3'):
                        time.sleep(0.15)  # Increased delay for servo stability
                    elif cmd.startswith('G1'):
                        time.sleep(0.05)
                    elif cmd.startswith('G0'):
                        time.sleep(0.02)
                    progress = (traveled / total_distance * 100) if total_distance > 0 else (i / len(commands) * 100)
                    progress_callback(progress, i)
        except Exception as e:
            logging.error(f"CNC error: {str(e)}")
            progress_callback(-1, 0)
            raise e

# GUI Application
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(PROJECT_NAME)
        self.geometry("800x600")
        self.style = ttk.Style()
        self.style.theme_use('clam')
        self.style.configure('TButton', background='lightblue', foreground='black', font=('Arial', 12), padding=10)
        self.style.map('TButton', background=[('active', 'blue')])
        self.style.configure('TLabel', font=('Arial', 12))
        self.style.configure('TProgressbar', thickness=20)
        self.current_frame = None
        self.capture = None
        self.selected_image = None
        self.sharpened = None
        self.segmented = None
        self.mask = None
        self.sketches = []
        self.selected_sketch = None
        self.selected_lines = None
        self.gcode_commands = None
        self.estimated_time = 0
        self.total_distance = 0
        self.camera_running = False
        self.print_thread = None
        self.runner = None
        self.current_command_idx = 0
        self.show_home()

    def add_header_footer(self, frame):
        header = ttk.Label(frame, text="MAGURA POLYTECHNIC INSTITUTE", font=("Arial", 18, "bold"))
        header.pack(pady=10)
        footer = ttk.Label(frame, text=THANKS, font=("Arial", 10))
        footer.pack(side="bottom", pady=10)
        return frame

    def switch_frame(self, new_frame):
        if self.current_frame:
            self.current_frame.destroy()
        self.current_frame = new_frame
        self.current_frame.pack(fill="both", expand=True)

    def show_home(self):
        frame = ttk.Frame(self)
        self.add_header_footer(frame)
        ttk.Label(frame, text=PROJECT_NAME, font=("Arial", 16)).pack(pady=10)
        ttk.Label(frame, text=DESCRIPTION, wraplength=700, justify="left").pack(pady=10)
        ttk.Label(frame, text=CREDITS, font=("Arial", 10)).pack(pady=10)
        ttk.Label(frame, text=COPYRIGHT, font=("Arial", 10)).pack(pady=10)
        ttk.Button(frame, text="Start New Portrait", command=self.show_camera).pack(pady=20)
        self.switch_frame(frame)

    def show_camera(self):
        frame = ttk.Frame(self)
        self.add_header_footer(frame)
        self.camera_label = ttk.Label(frame)
        self.camera_label.pack()
        ttk.Button(frame, text="Open Camera", command=self.open_camera).pack(pady=10)
        self.capture_button = ttk.Button(frame, text="Capture", command=self.capture_images, state="disabled")
        self.capture_button.pack(pady=10)
        self.thumbnail_frames = ttk.Frame(frame)
        self.thumbnail_frames.pack()
        self.switch_frame(frame)

    def open_camera(self):
        if self.capture:
            return
        self.capture = SmartCapture(CAMERA_PORT)
        self.capture_button["state"] = "normal"
        self.camera_running = True
        threading.Thread(target=self.update_camera, daemon=True).start()

    def update_camera(self):
        while self.camera_running:
            ret, frame = self.capture.cap.read()
            if ret and frame is not None:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(cv2.resize(frame, (640, 360)))
                imgtk = ImageTk.PhotoImage(image=img)
                self.camera_label.config(image=imgtk)
                self.camera_label.imgtk = imgtk
            time.sleep(0.03)

    def capture_images(self):
        self.capture_button["state"] = "disabled"
        progress = ttk.Progressbar(self.current_frame, mode="indeterminate")
        progress.pack(pady=10)
        progress.start()
        def capture_task():
            frames, _ = self.capture.capture_frames()
            self.after(0, lambda: self.process_captured_frames(frames, progress))
        threading.Thread(target=capture_task, daemon=True).start()

    def process_captured_frames(self, frames, progress):
        progress.stop()
        progress.destroy()
        self.camera_running = False
        self.capture.release()
        self.capture = None
        for widget in self.thumbnail_frames.winfo_children():
            widget.destroy()
        self.thumb_refs = []
        for i, frame in enumerate(frames):
            thumb = cv2.resize(frame, (200, 150))
            thumb = cv2.cvtColor(thumb, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(thumb)
            imgtk = ImageTk.PhotoImage(image=img)
            label = ttk.Label(self.thumbnail_frames, image=imgtk)
            label.grid(row=0, column=i, padx=10)
            label.imgtk = imgtk
            self.thumb_refs.append(imgtk)
            btn = ttk.Button(self.thumbnail_frames, text="Choose", command=lambda f=frame: self.choose_image(f))
            btn.grid(row=1, column=i)

    def choose_image(self, image):
        self.selected_image = image
        self.show_background_removal()

    def show_background_removal(self):
        frame = ttk.Frame(self)
        self.add_header_footer(frame)
        bg_frame = ttk.Frame(frame)
        bg_frame.pack(side="left")
        mask_frame = ttk.Frame(frame)
        mask_frame.pack(side="right")
        self.bg_label = ttk.Label(bg_frame)
        self.bg_label.pack()
        self.mask_label = ttk.Label(mask_frame)
        self.mask_label.pack()
        ttk.Button(frame, text="Redo (Few Iterations)", command=lambda: self.remove_bg(3)).pack(pady=10)
        ttk.Button(frame, text="Redo (Many Iterations)", command=lambda: self.remove_bg(7)).pack(pady=10)
        ttk.Button(frame, text="Confirm", command=self.confirm_bg).pack(pady=10)
        self.switch_frame(frame)
        self.remove_bg()

    def remove_bg(self, iterations=5):
        progress = ttk.Progressbar(self.current_frame, mode="indeterminate")
        progress.pack(pady=10)
        progress.start()
        def bg_task():
            temp_capture = SmartCapture()
            self.sharpened = temp_capture.sharpen_frame(self.selected_image)
            self.segmented, self.mask = temp_capture.remove_background(self.sharpened, iterations)
            temp_capture.release()
            self.after(0, lambda: self.update_bg_images(progress))
        threading.Thread(target=bg_task, daemon=True).start()

    def update_bg_images(self, progress):
        progress.stop()
        progress.destroy()
        self.display_image(self.segmented, self.bg_label)
        self.display_image(self.mask, self.mask_label)

    def confirm_bg(self):
        self.show_sketch_generation()

    def show_sketch_generation(self):
        frame = ttk.Frame(self)
        self.add_header_footer(frame)
        self.sketch_thumbs = ttk.Frame(frame)
        self.sketch_thumbs.pack()
        progress = ttk.Progressbar(frame, mode="indeterminate")
        progress.pack(pady=10)
        progress.start()
        def sketch_task():
            variants = [
                ("Low Sensitivity", RobustSketchGenerator(canny_low=150, canny_high=300, min_area=100, dilate_iterations=0)),
                ("Medium Sensitivity", RobustSketchGenerator(canny_low=100, canny_high=250, min_area=50, dilate_iterations=1)),
                ("High Sensitivity", RobustSketchGenerator(canny_low=50, canny_high=200, min_area=20, dilate_iterations=2))
            ]
            self.sketches = []
            self.sketch_refs = []
            for i, (name, gen) in enumerate(variants):
                sketch, lines, num_lines = gen.generate_sketch(self.segmented)
                est_time = num_lines * 0.7
                self.sketches.append((sketch, lines, num_lines, est_time))
            self.after(0, lambda: self.update_sketches(variants, progress))

        threading.Thread(target=sketch_task, daemon=True).start()

    def update_sketches(self, variants, progress):
        progress.stop()
        progress.destroy()
        for i in range(len(variants)):
            sketch, lines, num_lines, est_time = self.sketches[i]
            name = variants[i][0]
            thumb = cv2.resize(sketch, (200, 150))
            if len(thumb.shape) == 2:
                thumb = cv2.cvtColor(thumb, cv2.COLOR_GRAY2RGB)
            img = Image.fromarray(thumb)
            imgtk = ImageTk.PhotoImage(image=img)
            label = ttk.Label(self.sketch_thumbs, image=imgtk)
            label.grid(row=0, column=i, padx=10)
            label.imgtk = imgtk
            self.sketch_refs.append(imgtk)
            ttk.Label(self.sketch_thumbs, text=f"{name}\nLines: {num_lines}\nEst Time: {est_time:.1f}s").grid(row=1, column=i)
            btn = ttk.Button(self.sketch_thumbs, text="Choose", command=lambda s=sketch, l=lines: self.choose_sketch(s, l))
            btn.grid(row=2, column=i)

    def choose_sketch(self, sketch, lines):
        self.selected_sketch = sketch
        self.selected_lines = lines
        self.show_gcode_preview()

    def show_gcode_preview(self):
        frame = ttk.Frame(self)
        self.add_header_footer(frame)
        ttk.Label(frame, text="G-Code Preview").pack(pady=10)
        self.gcode_commands, self.estimated_time, self.total_distance = generate_gcode_from_sketch(self.selected_sketch, self.selected_lines)
        ttk.Label(frame, text=f"Estimated Draw Time: {self.estimated_time:.1f} seconds").pack(pady=10)
        gcode_text = scrolledtext.ScrolledText(frame, height=10, width=80)
        gcode_text.pack(pady=10)
        gcode_text.insert(tk.END, '\n'.join(cmd for cmd, _ in self.gcode_commands))
        gcode_text.config(state='disabled')
        ttk.Button(frame, text="Print", command=self.show_printing).pack(pady=20)
        self.switch_frame(frame)

    def show_printing(self):
        frame = ttk.Frame(self)
        self.add_header_footer(frame)
        ttk.Label(frame, text="Printing Setup").pack(pady=10)
        ttk.Label(frame, text="Manual Step: Jog to bottom-left, run G92 X0 Y0 in UGS, then close UGS to free port.").pack(pady=10)
        self.progress_bar = ttk.Progressbar(frame, maximum=100, value=0)
        self.progress_bar.pack(pady=10)
        self.start_button = ttk.Button(frame, text="Start Printing", command=self.start_print)
        self.start_button.pack(pady=5)
        self.pause_button = ttk.Button(frame, text="Pause", command=self.pause_print, state="disabled")
        self.pause_button.pack(pady=5)
        self.resume_button = ttk.Button(frame, text="Resume", command=self.resume_print, state="disabled")
        self.resume_button.pack(pady=5)
        self.cancel_button = ttk.Button(frame, text="Cancel", command=self.cancel_print, state="disabled")
        self.cancel_button.pack(pady=5)
        self.switch_frame(frame)

    def start_print(self):
        self.runner = CNCRunner()
        self.current_command_idx = 0
        self.start_button["state"] = "disabled"
        self.pause_button["state"] = "normal"
        self.cancel_button["state"] = "normal"
        self.print_thread = threading.Thread(target=self.print_task, daemon=True)
        self.print_thread.start()

    def print_task(self):
        try:
            self.runner.run(self.gcode_commands, self.update_progress, self.total_distance, self.current_command_idx)
            self.after(0, lambda: messagebox.showinfo("Success", "Drawing complete!"))
        except Exception as e:
            err_msg = str(e)
            self.after(0, lambda: messagebox.showerror("Error", err_msg))
        finally:
            self.after(0, self.reset_print_buttons)

    def pause_print(self):
        if self.runner:
            self.runner.paused = True
            self.pause_button["state"] = "disabled"
            self.resume_button["state"] = "normal"

    def resume_print(self):
        if self.runner:
            self.runner.paused = False
            self.pause_button["state"] = "normal"
            self.resume_button["state"] = "disabled"

    def cancel_print(self):
        if self.runner:
            self.runner.cancelled = True
            self.reset_print_buttons()

    def reset_print_buttons(self):
        self.start_button["state"] = "normal"
        self.pause_button["state"] = "disabled"
        self.resume_button["state"] = "disabled"
        self.cancel_button["state"] = "disabled"

    def update_progress(self, value, idx):
        self.current_command_idx = idx
        if value == -1:
            self.progress_bar["value"] = 0
        else:
            self.progress_bar["value"] = value

    def display_image(self, img, label):
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img, (300, 300))
        pil_img = Image.fromarray(img_resized)
        imgtk = ImageTk.PhotoImage(image=pil_img)
        label.config(image=imgtk)
        label.imgtk = imgtk

if __name__ == "__main__":
    app = App()
    app.mainloop()