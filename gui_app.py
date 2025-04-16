import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image # No need for ImageTk explicitly with CTkImage
import subprocess
import os
import threading
import json # Import json for parsing metrics

# Appearance settings
ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")

class TrafficDensityApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Traffic Density Detector")
        self.geometry("1100x750") # Slightly larger window

        # --- Variables ---
        self.input_image_path = ctk.StringVar()
        self.output_dir = ctk.StringVar(value=os.path.abspath("./results")) # Use absolute path for clarity
        self.model_path = ctk.StringVar(value="yolov8n.pt")
        self.confidence = ctk.DoubleVar(value=0.25)
        self.device = ctk.StringVar(value="cpu")
        self.status_text = ctk.StringVar(value="Status: Ready. Select an image.")
        self.metrics_text = ctk.StringVar(value="Metrics: N/A")
        self.output_image_path = None # Store the path to the output image

        # --- Layout Configuration ---
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1) # Image rows
        self.grid_rowconfigure(1, weight=0) # Config row
        self.grid_rowconfigure(2, weight=0) # Status row
        self.grid_rowconfigure(3, weight=0) # Progress bar row

        # --- Input Frame ---
        self.input_frame = ctk.CTkFrame(self, corner_radius=10)
        self.input_frame.grid(row=0, column=0, padx=15, pady=(15, 5), sticky="nsew")
        self.input_frame.grid_rowconfigure(1, weight=1)
        self.input_frame.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(self.input_frame, text="Input Image", font=ctk.CTkFont(weight="bold")).grid(row=0, column=0, padx=10, pady=(10, 5))
        self.input_image_display = ctk.CTkLabel(self.input_frame, text="Select an image using the button below", corner_radius=5)
        self.input_image_display.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")
        self.select_image_button = ctk.CTkButton(self.input_frame, text="Select Input Image", command=self.select_image)
        self.select_image_button.grid(row=2, column=0, padx=10, pady=(5, 10))

        # --- Output Frame ---
        self.output_frame = ctk.CTkFrame(self, corner_radius=10)
        self.output_frame.grid(row=0, column=1, padx=15, pady=(15, 5), sticky="nsew")
        self.output_frame.grid_rowconfigure(1, weight=1)
        self.output_frame.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(self.output_frame, text="Output Image & Metrics", font=ctk.CTkFont(weight="bold")).grid(row=0, column=0, padx=10, pady=(10, 5))
        self.output_image_display = ctk.CTkLabel(self.output_frame, text="Detection results will appear here", corner_radius=5)
        self.output_image_display.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")
        self.metrics_label = ctk.CTkLabel(self.output_frame, textvariable=self.metrics_text, justify="left")
        self.metrics_label.grid(row=2, column=0, padx=10, pady=(5, 10), sticky="ew")

        # --- Configuration Frame ---
        self.config_frame = ctk.CTkFrame(self, corner_radius=10)
        self.config_frame.grid(row=1, column=0, columnspan=2, padx=15, pady=10, sticky="ew")
        # Configure columns for alignment
        self.config_frame.grid_columnconfigure(0, weight=0) # Labels
        self.config_frame.grid_columnconfigure(1, weight=1) # Entries/Sliders
        self.config_frame.grid_columnconfigure(2, weight=0) # Buttons/Values
        self.config_frame.grid_columnconfigure(3, weight=0) # Labels
        self.config_frame.grid_columnconfigure(4, weight=1) # Entries/Menus
        self.config_frame.grid_columnconfigure(5, weight=0) # Buttons
        self.config_frame.grid_columnconfigure(6, weight=2) # Spacer/Run Button

        # Row 1: Output Dir, Model
        ctk.CTkLabel(self.config_frame, text="Output Dir:").grid(row=0, column=0, padx=(10, 2), pady=8, sticky="w")
        self.output_entry = ctk.CTkEntry(self.config_frame, textvariable=self.output_dir)
        self.output_entry.grid(row=0, column=1, padx=(0, 5), pady=8, sticky="ew")
        self.output_button = ctk.CTkButton(self.config_frame, text="Browse...", width=80, command=self.select_output_dir)
        self.output_button.grid(row=0, column=2, padx=(0, 20), pady=8, sticky="w")

        ctk.CTkLabel(self.config_frame, text="Model:").grid(row=0, column=3, padx=(10, 2), pady=8, sticky="w")
        self.model_entry = ctk.CTkEntry(self.config_frame, textvariable=self.model_path)
        self.model_entry.grid(row=0, column=4, padx=(0, 5), pady=8, sticky="ew")
        self.model_button = ctk.CTkButton(self.config_frame, text="Browse...", width=80, command=self.select_model)
        self.model_button.grid(row=0, column=5, padx=(0, 20), pady=8, sticky="w")

        # Row 2: Confidence, Device
        ctk.CTkLabel(self.config_frame, text="Confidence:").grid(row=1, column=0, padx=(10, 2), pady=8, sticky="w")
        self.conf_slider = ctk.CTkSlider(self.config_frame, from_=0.0, to=1.0, variable=self.confidence, command=lambda v: self.conf_label_val.configure(text=f"{v:.2f}"))
        self.conf_slider.grid(row=1, column=1, padx=(0, 10), pady=8, sticky="ew")
        self.conf_label_val = ctk.CTkLabel(self.config_frame, text=f"{self.confidence.get():.2f}", width=40)
        self.conf_label_val.grid(row=1, column=2, padx=(0, 20), pady=8, sticky="w")

        ctk.CTkLabel(self.config_frame, text="Device:").grid(row=1, column=3, padx=(10, 2), pady=8, sticky="w")
        self.device_options = ctk.CTkOptionMenu(self.config_frame, variable=self.device, values=["cpu", "0", "1", "2"])
        self.device_options.grid(row=1, column=4, columnspan=2, padx=(0, 20), pady=8, sticky="w")

        # Run Button (Spans both rows, aligned right)
        self.run_button = ctk.CTkButton(self.config_frame, text="Run Detection", command=self.start_detection_thread, height=50, font=ctk.CTkFont(weight="bold"))
        self.run_button.grid(row=0, column=6, rowspan=2, padx=15, pady=10, sticky="e")

        # --- Status Bar ---
        self.status_bar = ctk.CTkLabel(self, textvariable=self.status_text, anchor="w")
        self.status_bar.grid(row=2, column=0, columnspan=2, padx=15, pady=(5, 0), sticky="ew")

        # --- Progress Bar ---
        self.progress_bar = ctk.CTkProgressBar(self, mode='indeterminate')
        # Initially hidden by not placing it in the grid

    def select_image(self):
        path = filedialog.askopenfilename(
            title="Select Input Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.webp"), ("All files", "*.*")]
        )
        if path:
            self.input_image_path.set(path)
            self.status_text.set(f"Selected: {os.path.basename(path)}")
            self.display_image(path, self.input_image_display, is_input=True)
            # Clear previous output
            self.output_image_display.configure(image=None, text="Detection results will appear here")
            self.metrics_text.set("Metrics: N/A")
            self.output_image_path = None

    def select_output_dir(self):
        path = filedialog.askdirectory(title="Select Output Directory")
        if path:
            self.output_dir.set(path)
            self.status_text.set(f"Output directory set to: {path}")

    def select_model(self):
        path = filedialog.askopenfilename(
            title="Select YOLO Model",
            filetypes=[("PyTorch models", "*.pt"), ("All files", "*.*")]
        )
        if path:
            self.model_path.set(path)
            self.status_text.set(f"Model set to: {os.path.basename(path)}")

    def display_image(self, path, display_widget, is_input=False):
        try:
            self.update_idletasks() # Ensure geometry is up-to-date
            max_width = display_widget.winfo_width() - 20
            max_height = display_widget.winfo_height() - 40 # Allow space for label/button

            if max_width <= 1 or max_height <= 1:
                 max_width, max_height = 450, 450 # Adjusted default size

            img = Image.open(path)
            img.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)

            ctk_img = ctk.CTkImage(light_image=img, dark_image=img, size=img.size)
            display_widget.configure(image=ctk_img, text="")
            display_widget.image = ctk_img # Keep reference

            if not is_input:
                self.output_image_path = path # Store path if it's the output

        except Exception as e:
            error_text = f"Error loading image '{os.path.basename(path)}':\n{e}"
            self.status_text.set(f"Error: {error_text}")
            display_widget.configure(image=None, text=error_text)
            if not is_input:
                self.output_image_path = None

    def start_detection_thread(self):
        if not self.input_image_path.get():
            messagebox.showerror("Input Error", "Please select an input image first.")
            self.status_text.set("Status: Error - No input image selected.")
            return

        # Show and start progress bar
        self.progress_bar.grid(row=3, column=0, columnspan=2, padx=15, pady=(5, 10), sticky="ew")
        self.progress_bar.start()

        self.run_button.configure(state="disabled", text="Running...")
        self.status_text.set("Status: Processing... Please wait.")
        self.metrics_text.set("Metrics: Running...")
        self.output_image_display.configure(image=None, text="Processing...")
        self.output_image_path = None

        thread = threading.Thread(target=self.run_detection, daemon=True)
        thread.start()

    def run_detection(self):
        input_img = self.input_image_path.get()
        output_dir = self.output_dir.get()
        model = self.model_path.get()
        conf = self.confidence.get()
        device = self.device.get()
        script_path = os.path.abspath(os.path.join("AI-PROJECT", "traffic_density.py")) # Absolute path for script

        if not os.path.exists(script_path):
            self.handle_error(f"Detection script not found at {script_path}", "Script Not Found")
            return

        os.makedirs(output_dir, exist_ok=True)
        command = [
            "python3", script_path,
            "--image", input_img, "--output", output_dir,
            "--model", model, "--conf", str(conf), "--device", device,
            "--json_output" # ADD THIS ARGUMENT TO REQUEST JSON OUTPUT FROM SCRIPT
        ]

        try:
            print(f"Running command: {' '.join(command)}") # Log command
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=os.path.dirname(script_path)) # Run from script dir
            stdout, stderr = process.communicate()
            print("--- Script stdout ---")
            print(stdout)
            print("--- Script stderr ---")
            print(stderr)
            print("---------------------")

            if process.returncode == 0:
                # Try to parse metrics from the LAST line of stdout (assuming JSON)
                parsed_metrics = None
                if stdout:
                    try:
                        # Assume the last non-empty line is the JSON
                        last_line = stdout.strip().split('\n')[-1]
                        parsed_metrics = json.loads(last_line)
                    except json.JSONDecodeError as json_e:
                        print(f"Warning: Could not parse JSON from stdout last line: {json_e}")
                    except IndexError:
                         print("Warning: Stdout was empty or had no lines.")

                # Construct expected output path
                base_name = os.path.basename(input_img)
                name, _ = os.path.splitext(base_name)
                expected_output_image = os.path.join(output_dir, f"{name}_result.jpg") # Default assumption

                # If metrics dict contains an image path, use it!
                actual_output_image = parsed_metrics.get('output_image_path', expected_output_image) if parsed_metrics else expected_output_image

                if os.path.exists(actual_output_image):
                    self.status_text.set("Status: Detection complete.")
                    self.display_image(actual_output_image, self.output_image_display)

                    if parsed_metrics:
                        count = parsed_metrics.get('vehicle_count', 'N/A')
                        density_pct = parsed_metrics.get('density_percentage', 'N/A')
                        density_cls = parsed_metrics.get('density_class', 'N/A')
                        if isinstance(density_pct, (int, float)):
                             density_pct_str = f"{density_pct:.2f}%"
                        else:
                             density_pct_str = 'N/A'
                        self.metrics_text.set(f"Metrics: Count={count}, Density={density_pct_str}, Class={density_cls}")
                    else:
                        self.metrics_text.set("Metrics: Count=N/A, Density=N/A, Class=N/A (No metrics data received)")
                else:
                    self.handle_error(f"Script finished, but output image not found at {actual_output_image}", "Output Not Found")

            else:
                self.handle_error(f"Script failed (code {process.returncode}).\nStderr: {stderr}", "Script Error")

        except FileNotFoundError:
            self.handle_error(f"Python interpreter or script '{script_path}' not found. Check PATH.", "Execution Error")
        except Exception as e:
            self.handle_error(f"An unexpected error occurred: {e}", "Unexpected Error")
        finally:
            self.progress_bar.stop()
            self.progress_bar.grid_forget() # Hide progress bar
            self.run_button.configure(state="normal", text="Run Detection")

    def handle_error(self, message, title="Error"):
        print(f"Error: {message}")
        self.status_text.set(f"Status: Error - {title}")
        self.metrics_text.set("Metrics: Failed")
        self.output_image_display.configure(image=None, text=f"Error: {title}")
        self.output_image_path = None
        messagebox.showerror(title, message)
        # Ensure UI updates in the main thread if called from thread
        self.after(0, self._finalize_error_ui)

    def _finalize_error_ui(self):
        # This runs in the main thread to safely update UI after error
        self.progress_bar.stop()
        self.progress_bar.grid_forget()
        self.run_button.configure(state="normal", text="Run Detection")


if __name__ == "__main__":
    # Ensure the script is run from the workspace root or adjust paths accordingly
    print(f"Current working directory: {os.getcwd()}")
    app = TrafficDensityApp()
    app.mainloop() 