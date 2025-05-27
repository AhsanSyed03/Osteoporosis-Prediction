import os
import torch
import pandas as pd
import subprocess
from tkinter import Tk, filedialog
from main import run_prediction

# Define global paths
OUTPUT_DIR = r"C:\Users\ahsan\Desktop\Osteo_IV_prj\outputs"
CSV_PATH = os.path.join(OUTPUT_DIR, "powerbi_results.csv")
POWERBI_PATH = r"D:\Installed Application\PowerBI\bin\PBIDesktop.exe"
PBIX_PATH = os.path.join(OUTPUT_DIR, "OsteoReport.pbix")

def reset_csv():
    # If CSV exists, remove it (overwrite behavior)
    if os.path.exists(CSV_PATH):
        os.remove(CSV_PATH)
        print("[INFO] Old CSV file removed.")

def save_results_to_csv(image_path):
    if not os.path.exists(image_path):
        print(f"[ERROR] Image path does not exist: {image_path}")
        return False

    result = run_prediction(image_path)
    result["image"] = os.path.basename(image_path)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = pd.DataFrame([result])
    df.to_csv(CSV_PATH, index=False)
    print(f"[✅] Results saved to {CSV_PATH}")
    return True

def select_image_and_run():
    root = Tk()
    root.withdraw()

    file_path = filedialog.askopenfilename(
        title="Select an image for analysis",
        filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp *.tiff")]
    )

    if file_path:
        reset_csv()  # Delete previous CSV before running
        success = save_results_to_csv(file_path)
        if success:
            open_powerbi_report()
    else:
        print("[INFO] No file selected.")

def open_powerbi_report():
    if os.path.exists(PBIX_PATH):
        subprocess.Popen([POWERBI_PATH, PBIX_PATH])
    else:
        print(f"[ERROR] Power BI report not found at: {PBIX_PATH}")

if __name__ == "__main__":
    select_image_and_run()
