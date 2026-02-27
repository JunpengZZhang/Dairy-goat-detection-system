import tkinter as tk
from pathlib import Path
from threading import Thread
from tkinter import filedialog, messagebox, ttk

import cv2
from PIL import Image, ImageTk

from ultralytics import YOLO


class GoatBehaviorDetectionApp:
    """奶山羊行为检测桌面界面。."""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("奶山羊行为检测系统（YOLO11n）")
        self.root.geometry("1260x800")
        self.root.minsize(1020, 700)
        self.root.configure(bg="#f1f5f9")

        self.model = YOLO("yolo11n.pt")
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
        self.current_video_capture = None
        self.video_job = None

        self.preview_image = None

        self._setup_styles()
        self._build_layout()

    def _setup_styles(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TFrame", background="#f1f5f9")
        style.configure("Card.TFrame", background="#ffffff")
        style.configure("TLabel", background="#f1f5f9", foreground="#1e293b")
        style.configure("Title.TLabel", background="#f1f5f9", font=("Microsoft YaHei UI", 18, "bold"))
        style.configure("Section.TLabel", background="#ffffff", font=("Microsoft YaHei UI", 12, "bold"))
        style.configure("Subtle.TLabel", background="#ffffff", foreground="#64748b")
        style.configure("TLabelframe", background="#ffffff", borderwidth=0)
        style.configure("TLabelframe.Label", background="#ffffff", foreground="#334155", font=("Microsoft YaHei UI", 10, "bold"))
        style.configure("TButton", padding=8)
        style.configure("Primary.TButton", foreground="#ffffff", background="#2563eb")
        style.map("Primary.TButton", background=[("active", "#1d4ed8"), ("disabled", "#94a3b8")])

    def _build_layout(self):
        self.root.columnconfigure(0, weight=2)
        self.root.columnconfigure(1, weight=3)
        self.root.rowconfigure(0, weight=1)

        control_frame = ttk.Frame(self.root, padding=16, style="Card.TFrame")
        control_frame.grid(row=0, column=0, sticky="nsew")
        preview_frame = ttk.Frame(self.root, padding=16, style="Card.TFrame")
        preview_frame.grid(row=0, column=1, sticky="nsew")

        control_frame.columnconfigure(1, weight=1)
        preview_frame.rowconfigure(1, weight=1)
        preview_frame.columnconfigure(0, weight=1)

        title = ttk.Label(control_frame, text="奶山羊行为检测", style="Title.TLabel")
        title.grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 12))

        ttk.Label(control_frame, text="模型：YOLO11n · 支持图片/视频/摄像头实时检测", style="Subtle.TLabel").grid(
            row=1, column=0, columnspan=3, sticky="w", pady=(0, 10)
        )

        ttk.Label(control_frame, text="输入模式：").grid(row=2, column=0, sticky="w", pady=5)
        mode_box = ttk.Combobox(control_frame, textvariable=self.source_mode, values=["文件", "摄像头"], state="readonly")
        mode_box.grid(row=2, column=1, sticky="ew", pady=5)
        mode_box.bind("<<ComboboxSelected>>", lambda _: self._toggle_source_mode())
        ttk.Label(control_frame, text="").grid(row=2, column=2)

        ttk.Label(control_frame, text="输入源：").grid(row=3, column=0, sticky="w", pady=5)
        self.source_entry = ttk.Entry(control_frame, textvariable=self.source_path)
        self.source_entry.grid(row=3, column=1, sticky="ew", pady=5)
        self.choose_file_btn = ttk.Button(control_frame, text="选择文件", command=self._choose_source)
        self.choose_file_btn.grid(
            row=3, column=2, padx=(8, 0), pady=5
        )

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

        self.run_button = ttk.Button(action_frame, text="开始检测", style="Primary.TButton", command=self._start_detection)
        self.run_button.grid(row=0, column=0, sticky="ew", padx=(0, 6))
        self.stop_button = ttk.Button(action_frame, text="停止摄像头检测", command=self._stop_camera_detection, state="disabled")
        self.stop_button.grid(row=0, column=1, sticky="ew", padx=(6, 0))

        self.progress = ttk.Progressbar(control_frame, mode="indeterminate")
        self.progress.grid(row=10, column=0, columnspan=3, sticky="ew", pady=(10, 0))

        ttk.Label(preview_frame, text="结果预览", style="Section.TLabel").grid(
            row=0, column=0, sticky="w"
        )
        self.preview_label = ttk.Label(preview_frame, text="检测完成后显示结果图像/视频", anchor="center", relief="solid", background="#e2e8f0")
        self.preview_label.grid(row=1, column=0, sticky="nsew", pady=(8, 12))

        ttk.Label(preview_frame, text="运行日志", style="Section.TLabel").grid(
            row=2, column=0, sticky="w"
        )
        self.log_text = tk.Text(preview_frame, height=10, wrap="word")
        self.log_text.grid(row=3, column=0, sticky="nsew", pady=(6, 0))
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
        if self.source_mode.get() == "文件" and not self.source_path.get().strip():
            messagebox.showwarning("缺少输入", "请先选择图片或视频文件。")
            return

        self.run_button.configure(state="disabled")
        self.stop_button.configure(state="normal" if self.source_mode.get() == "摄像头" else "disabled")
        self.progress.start(10)
        self._log("开始检测，请稍候...")
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

            output_path = Path(results[0].save_dir)
            self.root.after(0, lambda: self._on_detection_success(output_path))
        except Exception as exc:
            self.root.after(0, lambda: self._on_detection_error(exc))

    def _run_camera_detection(self, classes):
        self._log(f"正在打开摄像头 {self.camera_index.get()} 并执行同步实时检测...")
        stream = self.model.predict(
            source=self.camera_index.get(),
            imgsz=self.imgsz.get(),
            conf=self.conf.get(),
            iou=self.iou.get(),
            max_det=self.max_det.get(),
            device=self.device.get().strip(),
            classes=classes,
            show=False,
            stream=True,
            verbose=False,
        )
        self.camera_session = stream
        for result in stream:
            if not self.is_running:
                break
            plotted = result.plot(labels=self.show_labels.get(), conf=self.show_conf.get())
            rgb = cv2.cvtColor(plotted, cv2.COLOR_BGR2RGB)
            self.root.after(0, lambda frm=rgb: self._display_frame(frm))
        self.root.after(0, self._on_camera_detection_stopped)

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
        self._log("摄像头检测已停止。")

    def _stop_camera_detection(self):
        self.is_running = False

    def _on_detection_success(self, output_dir: Path):
        self.progress.stop()
        self.is_running = False
        self.run_button.configure(state="normal")
        self.stop_button.configure(state="disabled")
        self._log(f"检测完成，结果保存在：{output_dir}")

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
        self._log(f"检测失败：{exc}")
        messagebox.showerror("检测失败", str(exc))

    def _find_preview_file(self, output_dir: Path):
        media_exts = {".jpg", ".jpeg", ".png", ".bmp", ".mp4", ".avi", ".mov", ".mkv"}
        candidates = [p for p in output_dir.iterdir() if p.suffix.lower() in media_exts]
        if not candidates:
            return None
        return max(candidates, key=lambda p: p.stat().st_mtime)

    def _update_preview(self, image_path: Path):
        image = Image.open(image_path)
        image.thumbnail((680, 500))
        self.preview_image = ImageTk.PhotoImage(image)
        self.preview_label.configure(image=self.preview_image, text="")

    def _log(self, message: str):
        self.log_text.insert("end", f"{message}\n")
        self.log_text.see("end")

    def _toggle_source_mode(self):
        is_file = self.source_mode.get() == "文件"
        self.source_entry.configure(state="normal" if is_file else "disabled")
        self.choose_file_btn.configure(state="normal" if is_file else "disabled")
        self.camera_entry.configure(state="disabled" if is_file else "normal")

    def _play_video_preview(self, video_path: Path):
        if self.current_video_capture is not None:
            self.current_video_capture.release()
        if self.video_job:
            self.root.after_cancel(self.video_job)

        self.current_video_capture = cv2.VideoCapture(str(video_path))
        if not self.current_video_capture.isOpened():
            self.preview_label.configure(text="视频预览打开失败，请在输出目录查看。", image="")
            return
        self._read_video_frame()

    def _read_video_frame(self):
        if self.current_video_capture is None:
            return
        success, frame = self.current_video_capture.read()
        if not success:
            self.current_video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            success, frame = self.current_video_capture.read()
            if not success:
                return

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self._display_frame(rgb)
        self.video_job = self.root.after(33, self._read_video_frame)


if __name__ == "__main__":
    app_root = tk.Tk()
    GoatBehaviorDetectionApp(app_root)
    app_root.mainloop()
