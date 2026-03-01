import tkinter as tk
from collections import Counter
from os import name as os_name
from pathlib import Path
from threading import Thread
from time import perf_counter
from tkinter import filedialog, messagebox, ttk

import cv2
from PIL import Image, ImageTk

from ultralytics import YOLO


class GoatBehaviorDetectionApp:
    """奶山羊行为检测桌面界面。."""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("奶山羊行为检测系统")
        self.root.geometry("1260x780")
        self.root.minsize(1020, 700)
        self.root.configure(bg="#f3f6fb")

        self.model = YOLO("best.pt")
        self.source_mode = tk.StringVar(value="文件")
        self.source_path = tk.StringVar(value="")
        self.camera_index = tk.IntVar(value=0)
        self.output_root = tk.StringVar(value=str(Path.cwd() / "runs" / "goat_detect"))
        self.run_name = tk.StringVar(value="exp")

        self.imgsz = tk.IntVar(value=640)
        self.conf = tk.DoubleVar(value=0.25)
        self.iou = tk.DoubleVar(value=0.7)
        self.max_det = tk.IntVar(value=300)
        self.device = tk.StringVar(value="cpu")
        self.classes = tk.StringVar(value="")
        self.save_txt = tk.BooleanVar(value=False)
        self.save_conf = tk.BooleanVar(value=False)
        self.show_labels = tk.BooleanVar(value=True)
        self.show_conf = tk.BooleanVar(value=True)
        self.is_running = False
        self.camera_session = None
        self.camera_capture = None
        self.current_video_capture = None
        self.video_job = None
        self.video_paused = True
        self.video_total_frames = 0
        self.video_fps = 30
        self.video_updating_slider = False
        self.last_summary = ""

        self.preview_image = None

        self._setup_styles()
        self._build_layout()

    def _setup_styles(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TFrame", background="#f3f6fb")
        style.configure("Card.TFrame", background="#ffffff")
        style.configure("TLabel", background="#ffffff", foreground="#1e293b", font=("Microsoft YaHei UI", 10))
        style.configure(
            "Title.TLabel", background="#ffffff", foreground="#0f172a", font=("Microsoft YaHei UI", 20, "bold")
        )
        style.configure(
            "Badge.TLabel",
            background="#2563eb",
            foreground="#eff6ff",
            font=("Microsoft YaHei UI", 9, "bold"),
            padding=(10, 4),
        )
        style.configure(
            "Section.TLabel", background="#ffffff", foreground="#0f172a", font=("Microsoft YaHei UI", 12, "bold")
        )
        style.configure("Subtle.TLabel", background="#ffffff", foreground="#64748b", font=("Microsoft YaHei UI", 9))
        style.configure("TLabelframe", background="#ffffff", borderwidth=0)
        style.configure(
            "TLabelframe.Label",
            background="#ffffff",
            foreground="#334155",
            font=("Microsoft YaHei UI", 10, "bold"),
        )
        style.configure(
            "TEntry", fieldbackground="#f8fafc", bordercolor="#cbd5e1", lightcolor="#cbd5e1", darkcolor="#cbd5e1"
        )
        style.map(
            "TEntry",
            bordercolor=[("focus", "#2563eb")],
            lightcolor=[("focus", "#2563eb")],
            darkcolor=[("focus", "#2563eb")],
        )
        style.configure("TCombobox", fieldbackground="#f8fafc", bordercolor="#cbd5e1")
        style.map("TCombobox", bordercolor=[("focus", "#2563eb")])
        style.configure("TButton", padding=9, font=("Microsoft YaHei UI", 10, "bold"))
        style.configure("Primary.TButton", foreground="#ffffff", background="#2563eb")
        style.map("Primary.TButton", background=[("active", "#1d4ed8"), ("disabled", "#94a3b8")])
        style.configure("Secondary.TButton", foreground="#334155", background="#e2e8f0")
        style.map("Secondary.TButton", background=[("active", "#cbd5e1"), ("disabled", "#f1f5f9")])
        style.configure(
            "Accent.Horizontal.TProgressbar", troughcolor="#e2e8f0", background="#2563eb", bordercolor="#e2e8f0"
        )

    def _build_layout(self):
        self.root.columnconfigure(0, weight=2)
        self.root.columnconfigure(1, weight=3)
        self.root.rowconfigure(0, weight=1)

        left_shell = ttk.Frame(self.root, padding=14)
        left_shell.grid(row=0, column=0, sticky="nsew")
        right_shell = ttk.Frame(self.root, padding=(0, 14, 14, 14))
        right_shell.grid(row=0, column=1, sticky="nsew")

        control_frame = ttk.Frame(left_shell, padding=18, style="Card.TFrame")
        control_frame.pack(fill="both", expand=True)
        preview_frame = ttk.Frame(right_shell, padding=18, style="Card.TFrame")
        preview_frame.pack(fill="both", expand=True)

        control_frame.columnconfigure(1, weight=1)
        preview_frame.rowconfigure(1, weight=1)
        preview_frame.columnconfigure(0, weight=1)

        title = ttk.Label(control_frame, text="奶山羊行为检测", style="Title.TLabel")
        title.grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 12))

        ttk.Label(control_frame, text="智能识别", style="Badge.TLabel").grid(
            row=0, column=2, sticky="e", pady=(0, 12)
        )

        ttk.Label(control_frame, text="模型：YOLO11和ELSLowFast-LSTM · 支持图片/视频/摄像头实时检测", style="Subtle.TLabel").grid(
            row=1, column=0, columnspan=3, sticky="w", pady=(0, 10)
        )

        ttk.Label(control_frame, text="输入模式：").grid(row=2, column=0, sticky="w", pady=5)
        mode_box = ttk.Combobox(
            control_frame, textvariable=self.source_mode, values=["文件", "摄像头"], state="readonly"
        )
        mode_box.grid(row=2, column=1, sticky="ew", pady=5)
        mode_box.bind("<<ComboboxSelected>>", lambda _: self._toggle_source_mode())
        ttk.Label(control_frame, text="").grid(row=2, column=2)

        ttk.Label(control_frame, text="输入源：").grid(row=3, column=0, sticky="w", pady=5)
        self.source_entry = ttk.Entry(control_frame, textvariable=self.source_path)
        self.source_entry.grid(row=3, column=1, sticky="ew", pady=5)
        self.choose_file_btn = ttk.Button(control_frame, text="选择文件", command=self._choose_source)
        self.choose_file_btn.grid(row=3, column=2, padx=(8, 0), pady=5)

        ttk.Label(control_frame, text="摄像头编号：").grid(row=4, column=0, sticky="w", pady=5)
        self.camera_entry = ttk.Entry(control_frame, textvariable=self.camera_index)
        self.camera_entry.grid(row=4, column=1, sticky="ew", pady=5)

        ttk.Label(control_frame, text="输出目录：").grid(row=5, column=0, sticky="w", pady=5)
        ttk.Entry(control_frame, textvariable=self.output_root).grid(row=5, column=1, sticky="ew", pady=5)
        ttk.Button(control_frame, text="选择目录", command=self._choose_output).grid(
            row=5, column=2, padx=(8, 0), pady=5
        )

        ttk.Label(control_frame, text="运行名称：").grid(row=6, column=0, sticky="w", pady=5)
        ttk.Entry(control_frame, textvariable=self.run_name).grid(row=6, column=1, sticky="ew", pady=5)

        settings = ttk.LabelFrame(control_frame, text="检测参数", padding=10)
        settings.grid(row=7, column=0, columnspan=3, sticky="ew", pady=(12, 0))
        settings.columnconfigure(1, weight=1)

        self._add_setting_row(settings, "图像尺寸 imgsz", self.imgsz, 0)
        self._add_setting_row(settings, "置信度 conf", self.conf, 1)
        self._add_setting_row(settings, "IoU 阈值", self.iou, 2)
        self._add_setting_row(settings, "最大检测数", self.max_det, 3)
        self._add_setting_row(settings, "设备 device", self.device, 4)
        self._add_setting_row(settings, "类别 classes(逗号分隔)", self.classes, 5)

        options = ttk.LabelFrame(control_frame, text="保存与显示选项", padding=10)
        options.grid(row=8, column=0, columnspan=3, sticky="ew", pady=(12, 0))

        ttk.Checkbutton(options, text="保存 txt 标注", variable=self.save_txt).grid(
            row=0, column=0, sticky="w", padx=(0, 15), pady=4
        )
        ttk.Checkbutton(options, text="txt 含 conf", variable=self.save_conf).grid(row=0, column=1, sticky="w", pady=4)
        ttk.Checkbutton(options, text="显示标签", variable=self.show_labels).grid(
            row=1, column=0, sticky="w", padx=(0, 15), pady=4
        )
        ttk.Checkbutton(options, text="显示置信度", variable=self.show_conf).grid(row=1, column=1, sticky="w", pady=4)

        action_frame = ttk.Frame(control_frame, style="Card.TFrame")
        action_frame.grid(row=9, column=0, columnspan=3, sticky="ew", pady=(14, 0))
        action_frame.columnconfigure(0, weight=1)
        action_frame.columnconfigure(1, weight=1)

        self.run_button = ttk.Button(
            action_frame, text="开始检测", style="Primary.TButton", command=self._start_detection
        )
        self.run_button.grid(row=0, column=0, sticky="ew", padx=(0, 6))
        self.stop_button = ttk.Button(
            action_frame,
            text="停止摄像头检测",
            style="Secondary.TButton",
            command=self._stop_camera_detection,
            state="disabled",
        )
        self.stop_button.grid(row=0, column=1, sticky="ew", padx=(6, 0))

        self.progress = ttk.Progressbar(control_frame, mode="indeterminate", style="Accent.Horizontal.TProgressbar")
        self.progress.grid(row=10, column=0, columnspan=3, sticky="ew", pady=(10, 0))

        ttk.Label(preview_frame, text="结果预览", style="Section.TLabel").grid(row=0, column=0, sticky="w")
        self.preview_label = ttk.Label(
            preview_frame,
            text="检测完成后显示结果图像/视频",
            anchor="center",
            relief="flat",
            background="#eef3ee",
            foreground="#607269",
            padding=18,
        )
        self.preview_label.grid(row=1, column=0, sticky="nsew", pady=(8, 12))

        self.video_control_frame = ttk.Frame(preview_frame, style="Card.TFrame")
        self.video_control_frame.grid(row=2, column=0, sticky="ew", pady=(0, 10))
        self.video_control_frame.columnconfigure(1, weight=1)
        self.video_control_frame.grid_remove()

        self.play_pause_button = ttk.Button(
            self.video_control_frame,
            text="开始",
            style="Secondary.TButton",
            command=self._toggle_video_play_pause,
            state="disabled",
        )
        self.play_pause_button.grid(row=0, column=0, padx=(0, 8))
        self.video_progress = ttk.Scale(
            self.video_control_frame,
            from_=0,
            to=100,
            orient="horizontal",
            command=self._on_video_seek,
        )
        self.video_progress.grid(row=0, column=1, sticky="ew")
        self.video_progress.configure(state="disabled")

        ttk.Label(preview_frame, text="运行日志", style="Section.TLabel").grid(row=3, column=0, sticky="w")
        self.log_text = tk.Text(
            preview_frame,
            height=10,
            wrap="word",
            bg="#1c2a24",
            fg="#e7efe9",
            insertbackground="#e7efe9",
            relief="flat",
            padx=12,
            pady=10,
            font=("Consolas", 10),
        )
        self.log_text.grid(row=4, column=0, sticky="nsew", pady=(6, 0))

        ttk.Label(preview_frame, text="检测摘要", style="Section.TLabel").grid(row=5, column=0, sticky="w", pady=(10, 0))
        self.summary_text = tk.Text(
            preview_frame,
            height=7,
            wrap="word",
            bg="#f8fafc",
            fg="#0f172a",
            insertbackground="#0f172a",
            relief="flat",
            padx=12,
            pady=10,
            font=("Consolas", 10),
            state="disabled",
        )
        self.summary_text.grid(row=6, column=0, sticky="nsew", pady=(6, 0))

        summary_actions = ttk.Frame(preview_frame, style="Card.TFrame")
        summary_actions.grid(row=7, column=0, sticky="e", pady=(8, 0))
        ttk.Button(summary_actions, text="复制摘要", style="Secondary.TButton", command=self._copy_summary).pack()
        self._toggle_source_mode()

    def _add_setting_row(self, parent, label_text, variable, row):
        ttk.Label(parent, text=label_text).grid(row=row, column=0, sticky="w", pady=3)
        ttk.Entry(parent, textvariable=variable).grid(row=row, column=1, sticky="ew", pady=3)

    def _choose_source(self):
        filetypes = [
            ("支持的媒体", "*.jpg *.jpeg *.png *.bmp *.mp4 *.avi *.mov *.mkv"),
            ("所有文件", "*.*"),
        ]
        selected = filedialog.askopenfilename(title="选择图片或视频", filetypes=filetypes)
        if selected:
            self.source_path.set(selected)

    def _choose_output(self):
        selected = filedialog.askdirectory(title="选择输出目录")
        if selected:
            self.output_root.set(selected)

    def _parse_classes(self):
        raw = self.classes.get().strip()
        if not raw:
            return None
        return [int(x.strip()) for x in raw.split(",") if x.strip()]

    def _start_detection(self):
        self._stop_video_preview(reset_controls=True)
        if self.source_mode.get() == "文件" and not self.source_path.get().strip():
            messagebox.showwarning("缺少输入", "请先选择图片或视频文件。")
            return

        self.run_button.configure(state="disabled")
        self.stop_button.configure(state="normal" if self.source_mode.get() == "摄像头" else "disabled")
        self.progress.start(10)
        self._log("开始检测，请稍候...")
        self._set_summary("检测运行中，摘要将在检测结束后自动生成。")
        self.is_running = True
        Thread(target=self._run_detection, daemon=True).start()

    def _run_detection(self):
        try:
            classes = self._parse_classes()
            output_root = Path(self.output_root.get()).expanduser().resolve()
            output_root.mkdir(parents=True, exist_ok=True)

            if self.source_mode.get() == "摄像头":
                self._run_camera_detection(classes)
                return

            start_time = perf_counter()
            results = self.model.predict(
                source=self.source_path.get().strip(),
                imgsz=self.imgsz.get(),
                conf=self.conf.get(),
                iou=self.iou.get(),
                max_det=self.max_det.get(),
                device=self.device.get().strip(),
                classes=classes,
                save=True,
                save_txt=self.save_txt.get(),
                save_conf=self.save_conf.get(),
                show_labels=self.show_labels.get(),
                show_conf=self.show_conf.get(),
                project=str(output_root),
                name=self.run_name.get().strip() or "exp",
                exist_ok=True,
                verbose=False,
            )
            elapsed = perf_counter() - start_time
            summary = self._build_detection_summary(results, elapsed)

            output_path = Path(results[0].save_dir)
            self.root.after(0, lambda: self._on_detection_success(output_path, summary))
        except Exception as exc:
            self.root.after(0, lambda: self._on_detection_error(exc))

    def _run_camera_detection(self, classes):
        self._log(f"正在打开摄像头 {self.camera_index.get()} 并执行同步实时检测...")
        capture = self._open_camera_capture(self.camera_index.get())
        if not capture.isOpened():
            raise RuntimeError("摄像头打开失败，请检查编号是否正确或关闭占用摄像头的软件后重试。")

        self.camera_capture = capture
        failed_reads = 0
        try:
            while self.is_running:
                success, frame = capture.read()
                if not success:
                    failed_reads += 1
                    if failed_reads == 1:
                        self._log("摄像头读取帧失败，正在重试...")
                    if failed_reads >= 10:
                        raise RuntimeError("摄像头连续读取失败，已停止检测。")
                    continue

                failed_reads = 0
                result = self.model.predict(
                    source=frame,
                    imgsz=self.imgsz.get(),
                    conf=self.conf.get(),
                    iou=self.iou.get(),
                    max_det=self.max_det.get(),
                    device=self.device.get().strip(),
                    classes=classes,
                    show=False,
                    save=False,
                    verbose=False,
                )[0]
                plotted = result.plot(labels=self.show_labels.get(), conf=self.show_conf.get())
                rgb = cv2.cvtColor(plotted, cv2.COLOR_BGR2RGB)
                self.root.after(0, lambda frm=rgb: self._display_frame(frm))
        finally:
            self._release_camera_capture()
            self.root.after(0, self._on_camera_detection_stopped)

    def _open_camera_capture(self, camera_index: int):
        if os_name == "nt":
            dshow_capture = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
            if dshow_capture.isOpened():
                return dshow_capture
            dshow_capture.release()
        return cv2.VideoCapture(camera_index)

    def _release_camera_capture(self):
        if self.camera_capture is not None:
            self.camera_capture.release()
            self.camera_capture = None

    def _display_frame(self, frame_rgb):
        image = Image.fromarray(frame_rgb)
        image.thumbnail((760, 540))
        self.preview_image = ImageTk.PhotoImage(image)
        self.preview_label.configure(image=self.preview_image, text="")

    def _on_camera_detection_stopped(self):
        self.progress.stop()
        self.is_running = False
        self.run_button.configure(state="normal")
        self.stop_button.configure(state="disabled")
        self._set_summary("摄像头检测已停止。实时模式暂不生成自动统计摘要。")
        self._log("摄像头检测已停止。")

    def _stop_camera_detection(self):
        self.is_running = False
        self._release_camera_capture()

    def _on_detection_success(self, output_dir: Path, summary: str):
        self.progress.stop()
        self.is_running = False
        self.run_button.configure(state="normal")
        self.stop_button.configure(state="disabled")
        self._log(f"检测完成，结果保存在：{output_dir}")
        self._set_summary(summary)

        preview_file = self._find_preview_file(output_dir)
        if preview_file:
            if preview_file.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"}:
                self._play_video_preview(preview_file)
            else:
                self._update_preview(preview_file)
            self._log(f"预览文件：{preview_file.name}")
        else:
            self.preview_label.configure(text="未找到可预览文件，请在输出目录查看结果。", image="")

    def _on_detection_error(self, exc: Exception):
        self.progress.stop()
        self.is_running = False
        self.run_button.configure(state="normal")
        self.stop_button.configure(state="disabled")
        self._set_summary("检测失败，未生成摘要。\n请检查输入参数、模型路径和推理设备配置。")
        self._log(f"检测失败：{exc}")
        messagebox.showerror("检测失败", str(exc))

    def _find_preview_file(self, output_dir: Path):
        media_exts = {".jpg", ".jpeg", ".png", ".bmp", ".mp4", ".avi", ".mov", ".mkv"}
        candidates = [p for p in output_dir.iterdir() if p.suffix.lower() in media_exts]
        if not candidates:
            return None
        return max(candidates, key=lambda p: p.stat().st_mtime)

    def _update_preview(self, image_path: Path):
        self._stop_video_preview(reset_controls=True)
        image = Image.open(image_path)
        image.thumbnail((680, 500))
        self.preview_image = ImageTk.PhotoImage(image)
        self.preview_label.configure(image=self.preview_image, text="")

    def _log(self, message: str):
        self.log_text.insert("end", f"{message}\n")
        self.log_text.see("end")

    def _set_summary(self, summary: str):
        self.last_summary = summary
        self.summary_text.configure(state="normal")
        self.summary_text.delete("1.0", "end")
        self.summary_text.insert("1.0", summary)
        self.summary_text.configure(state="disabled")

    def _copy_summary(self):
        if not self.last_summary:
            messagebox.showinfo("无可复制内容", "请先运行一次检测以生成摘要。")
            return
        self.root.clipboard_clear()
        self.root.clipboard_append(self.last_summary)
        self._log("检测摘要已复制到剪贴板。")

    def _build_detection_summary(self, results, elapsed_seconds: float):
        if not results:
            return "未获得检测结果。"

        frame_count = len(results)
        class_counter = Counter()
        confidence_values = []

        for result in results:
            names = result.names
            for box in result.boxes:
                cls_id = int(box.cls.item())
                class_name = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else str(cls_id)
                class_counter[class_name] += 1
                confidence_values.append(float(box.conf.item()))

        total_detections = sum(class_counter.values())
        avg_conf = (sum(confidence_values) / len(confidence_values)) if confidence_values else 0.0
        fps = frame_count / elapsed_seconds if elapsed_seconds > 0 else 0.0

        class_lines = [f"- {name}: {count}" for name, count in class_counter.most_common()]
        class_block = "\n".join(class_lines) if class_lines else "- 无目标"

        return (
            "=== 奶山羊行为检测摘要 ===\n"
            f"处理帧数: {frame_count}\n"
            f"总检测目标数: {total_detections}\n"
            f"平均置信度: {avg_conf:.3f}\n"
            f"总耗时: {elapsed_seconds:.2f} 秒\n"
            f"处理速度: {fps:.2f} 帧/秒\n"
            "类别统计:\n"
            f"{class_block}"
        )

    def _toggle_source_mode(self):
        is_file = self.source_mode.get() == "文件"
        self.source_entry.configure(state="normal" if is_file else "disabled")
        self.choose_file_btn.configure(state="normal" if is_file else "disabled")
        self.camera_entry.configure(state="disabled" if is_file else "normal")
        if not is_file:
            self._stop_video_preview(reset_controls=True)

    def _play_video_preview(self, video_path: Path):
        self._stop_video_preview(reset_controls=False)
        self.current_video_capture = cv2.VideoCapture(str(video_path))
        if not self.current_video_capture.isOpened():
            self.preview_label.configure(text="视频预览打开失败，请在输出目录查看。", image="")
            return

        self.video_total_frames = int(self.current_video_capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.video_fps = self.current_video_capture.get(cv2.CAP_PROP_FPS) or 30
        if self.video_fps <= 0:
            self.video_fps = 30

        self.video_updating_slider = True
        self.video_progress.configure(to=max(self.video_total_frames - 1, 1), state="normal")
        self.video_progress.set(0)
        self.video_updating_slider = False

        self.video_paused = True
        self.play_pause_button.configure(text="开始", state="normal")
        self.video_control_frame.grid()
        self._read_video_frame()

    def _stop_video_preview(self, reset_controls: bool):
        if self.video_job:
            self.root.after_cancel(self.video_job)
            self.video_job = None
        if self.current_video_capture is not None:
            self.current_video_capture.release()
            self.current_video_capture = None
        self.video_paused = True
        if reset_controls:
            self.video_control_frame.grid_remove()
            self.play_pause_button.configure(text="开始", state="disabled")
            self.video_progress.configure(state="disabled", to=100)
            self.video_progress.set(0)

    def _toggle_video_play_pause(self):
        if self.current_video_capture is None:
            return
        self.video_paused = not self.video_paused
        self.play_pause_button.configure(text="暂停" if not self.video_paused else "开始")
        if not self.video_paused:
            self._schedule_next_video_frame()

    def _schedule_next_video_frame(self):
        if self.video_paused or self.current_video_capture is None:
            return
        delay = max(int(1000 / self.video_fps), 1)
        self.video_job = self.root.after(delay, self._read_video_frame)

    def _on_video_seek(self, value):
        if self.current_video_capture is None or self.video_updating_slider:
            return
        frame_index = int(float(value))
        self.current_video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        self._read_video_frame(seek_only=self.video_paused)

    def _read_video_frame(self, seek_only: bool = False):
        if self.current_video_capture is None:
            return
        success, frame = self.current_video_capture.read()
        if not success:
            self.video_paused = True
            self.play_pause_button.configure(text="开始")
            self.video_job = None
            return

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self._display_frame(rgb)
        current_frame = int(self.current_video_capture.get(cv2.CAP_PROP_POS_FRAMES))
        self.video_updating_slider = True
        self.video_progress.set(current_frame)
        self.video_updating_slider = False
        self.video_job = None
        if not seek_only:
            self._schedule_next_video_frame()


if __name__ == "__main__":
    app_root = tk.Tk()
    GoatBehaviorDetectionApp(app_root)
    app_root.mainloop()
