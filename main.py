from pathlib import Path
from threading import Thread
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageTk
from ultralytics import YOLO


class GoatBehaviorDetectionApp:
    """奶山羊行为检测桌面界面。"""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("奶山羊行为检测系统（YOLO11n）")
        self.root.geometry("1180x760")
        self.root.minsize(1020, 700)

        self.model = YOLO("yolo11n.pt")
        self.source_path = tk.StringVar(value="")
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

        self.preview_image = None

        self._build_layout()

    def _build_layout(self):
        self.root.columnconfigure(0, weight=2)
        self.root.columnconfigure(1, weight=3)
        self.root.rowconfigure(0, weight=1)

        control_frame = ttk.Frame(self.root, padding=16)
        control_frame.grid(row=0, column=0, sticky="nsew")
        preview_frame = ttk.Frame(self.root, padding=16)
        preview_frame.grid(row=0, column=1, sticky="nsew")

        control_frame.columnconfigure(1, weight=1)
        preview_frame.rowconfigure(1, weight=1)
        preview_frame.columnconfigure(0, weight=1)

        title = ttk.Label(control_frame, text="奶山羊行为检测", font=("Microsoft YaHei UI", 16, "bold"))
        title.grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 12))

        ttk.Label(control_frame, text="输入源：").grid(row=1, column=0, sticky="w", pady=5)
        ttk.Entry(control_frame, textvariable=self.source_path).grid(row=1, column=1, sticky="ew", pady=5)
        ttk.Button(control_frame, text="选择文件", command=self._choose_source).grid(row=1, column=2, padx=(8, 0), pady=5)

        ttk.Label(control_frame, text="输出目录：").grid(row=2, column=0, sticky="w", pady=5)
        ttk.Entry(control_frame, textvariable=self.output_root).grid(row=2, column=1, sticky="ew", pady=5)
        ttk.Button(control_frame, text="选择目录", command=self._choose_output).grid(row=2, column=2, padx=(8, 0), pady=5)

        ttk.Label(control_frame, text="运行名称：").grid(row=3, column=0, sticky="w", pady=5)
        ttk.Entry(control_frame, textvariable=self.run_name).grid(row=3, column=1, sticky="ew", pady=5)

        settings = ttk.LabelFrame(control_frame, text="检测参数", padding=10)
        settings.grid(row=4, column=0, columnspan=3, sticky="ew", pady=(12, 0))
        settings.columnconfigure(1, weight=1)

        self._add_setting_row(settings, "图像尺寸 imgsz", self.imgsz, 0)
        self._add_setting_row(settings, "置信度 conf", self.conf, 1)
        self._add_setting_row(settings, "IoU 阈值", self.iou, 2)
        self._add_setting_row(settings, "最大检测数", self.max_det, 3)
        self._add_setting_row(settings, "设备 device", self.device, 4)
        self._add_setting_row(settings, "类别 classes(逗号分隔)", self.classes, 5)

        options = ttk.LabelFrame(control_frame, text="保存与显示选项", padding=10)
        options.grid(row=5, column=0, columnspan=3, sticky="ew", pady=(12, 0))

        ttk.Checkbutton(options, text="保存 txt 标注", variable=self.save_txt).grid(row=0, column=0, sticky="w", padx=(0, 15), pady=4)
        ttk.Checkbutton(options, text="txt 含 conf", variable=self.save_conf).grid(row=0, column=1, sticky="w", pady=4)
        ttk.Checkbutton(options, text="显示标签", variable=self.show_labels).grid(row=1, column=0, sticky="w", padx=(0, 15), pady=4)
        ttk.Checkbutton(options, text="显示置信度", variable=self.show_conf).grid(row=1, column=1, sticky="w", pady=4)

        self.run_button = ttk.Button(control_frame, text="开始检测", command=self._start_detection)
        self.run_button.grid(row=6, column=0, columnspan=3, sticky="ew", pady=(14, 0))

        self.progress = ttk.Progressbar(control_frame, mode="indeterminate")
        self.progress.grid(row=7, column=0, columnspan=3, sticky="ew", pady=(10, 0))

        ttk.Label(preview_frame, text="结果预览", font=("Microsoft YaHei UI", 13, "bold")).grid(row=0, column=0, sticky="w")
        self.preview_label = ttk.Label(preview_frame, text="检测完成后显示结果图像", anchor="center", relief="solid")
        self.preview_label.grid(row=1, column=0, sticky="nsew", pady=(8, 12))

        ttk.Label(preview_frame, text="运行日志", font=("Microsoft YaHei UI", 12, "bold")).grid(row=2, column=0, sticky="w")
        self.log_text = tk.Text(preview_frame, height=10, wrap="word")
        self.log_text.grid(row=3, column=0, sticky="nsew", pady=(6, 0))

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
        if not self.source_path.get().strip():
            messagebox.showwarning("缺少输入", "请先选择图片或视频文件。")
            return

        self.run_button.configure(state="disabled")
        self.progress.start(10)
        self._log("开始检测，请稍候...")
        Thread(target=self._run_detection, daemon=True).start()

    def _run_detection(self):
        try:
            classes = self._parse_classes()
            output_root = Path(self.output_root.get()).expanduser().resolve()
            output_root.mkdir(parents=True, exist_ok=True)

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

    def _on_detection_success(self, output_dir: Path):
        self.progress.stop()
        self.run_button.configure(state="normal")
        self._log(f"检测完成，结果保存在：{output_dir}")

        preview_file = self._find_preview_file(output_dir)
        if preview_file:
            self._update_preview(preview_file)
            self._log(f"预览文件：{preview_file.name}")
        else:
            self.preview_label.configure(text="未找到可预览图片，请在输出目录查看视频结果。", image="")

    def _on_detection_error(self, exc: Exception):
        self.progress.stop()
        self.run_button.configure(state="normal")
        self._log(f"检测失败：{exc}")
        messagebox.showerror("检测失败", str(exc))

    def _find_preview_file(self, output_dir: Path):
        image_exts = {".jpg", ".jpeg", ".png", ".bmp"}
        candidates = [p for p in output_dir.iterdir() if p.suffix.lower() in image_exts]
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


if __name__ == "__main__":
    app_root = tk.Tk()
    GoatBehaviorDetectionApp(app_root)
    app_root.mainloop()
