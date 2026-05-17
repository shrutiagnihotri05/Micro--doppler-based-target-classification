import tkinter as tk
from tkinter import filedialog, messagebox
import customtkinter as ctk
import pandas as pd
import pickle
import webbrowser
from urllib.parse import quote

MODEL_PATH = "model.pkl"
SUPPORT_EMAIL = "fashionvilla0817@gmail.com"

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
except Exception as e:
    messagebox.showerror("Error", f"Failed to load model:\n{e}")
    exit()

def load_data(file_path):
    df = pd.read_csv(file_path, header=None)
    return df.iloc[0, :5000].values

def send_feedback_email(message_text):
    subject = quote("Micro-Doppler Feedback / Query")
    body = quote(message_text)
    mailto_link = f"mailto:{SUPPORT_EMAIL}?subject={subject}&body={body}"
    webbrowser.open(mailto_link)

def classify_signal():
    file_path = filedialog.askopenfilename(
        title="Select CSV File",
        filetypes=[("CSV Files", "*.csv")]
    )

    if not file_path:
        return

    try:
        status_label.configure(text="Analyzing Signal...", text_color="#F59E0B")
        app.update()

        data = load_data(file_path)

        if len(data) != 5000:
            messagebox.showerror("Invalid File", "Expected exactly 5000 features.")
            return

        prediction = model.predict(data.reshape(1, -1))

        if prediction[0] == 1:
            result_label.configure(text="DRONE DETECTED", text_color="#EF4444")
            icon_label.configure(text="🚁")
        else:
            result_label.configure(text="BIRD DETECTED", text_color="#10B981")
            icon_label.configure(text="🕊")

        status_label.configure(text="Classification Complete", text_color="#10B981")

    except Exception as e:
        messagebox.showerror("Error", str(e))
        status_label.configure(text="Error Occurred", text_color="#EF4444")

def open_feedback_window():
    feedback_window = ctk.CTkToplevel(app)
    feedback_window.title("Contact & Feedback")
    feedback_window.geometry("700x550")
    feedback_window.configure(fg_color="#0F172A")
    
    feedback_window.transient(app)
    feedback_window.grab_set()

    title = ctk.CTkLabel(feedback_window, text="Contact & Feedback", font=("Segoe UI", 30, "bold"))
    title.pack(pady=(25, 10))

    subtitle = ctk.CTkLabel(feedback_window, text="Write your feedback or query below", font=("Segoe UI", 14), text_color="#94A3B8")
    subtitle.pack()

    message_box = ctk.CTkTextbox(feedback_window, width=600, height=250, font=("Segoe UI", 14), corner_radius=15)
    message_box.pack(pady=30)

    def submit_feedback():
        message_text = message_box.get("1.0", "end").strip()
        if not message_text:
            messagebox.showwarning("Empty Message", "Please write your feedback.")
            return
        send_feedback_email(message_text)
        messagebox.showinfo("Email Opened", "Your email application has been opened.")
        feedback_window.destroy()

    submit_button = ctk.CTkButton(
        feedback_window, text="Send Feedback", command=submit_feedback,
        width=220, height=50, font=("Segoe UI", 15, "bold"), corner_radius=12,
        fg_color="#2563EB", hover_color="#1D4ED8"
    )
    submit_button.pack(pady=10)


app = ctk.CTk()
app.title("Micro-Doppler Classification System")

try:
    app.state('zoomed')
except tk.TclError:
    app.geometry("1024x768")

header = ctk.CTkFrame(app, fg_color="#111827", corner_radius=0)
header.pack(side="top", fill="x")

title = ctk.CTkLabel(header, text="Micro-Doppler Radar Intelligence Platform", font=("Segoe UI", 34, "bold"))
title.pack(pady=(25, 5))

subtitle = ctk.CTkLabel(header, text="AI Powered Drone vs Bird Classification Build Using Support Vector Machine (SVM)", font=("Segoe UI", 15), text_color="#94A3B8")
subtitle.pack(pady=(0, 20))


footer = ctk.CTkFrame(app, fg_color="#111827", corner_radius=0)
footer.pack(side="bottom", fill="x")

team_label = ctk.CTkLabel(footer, text="Damodar Barhate  •  Shruti Agnihotri  •  Ishika Jaiswal", font=("Segoe UI", 13), text_color="#CBD5E1")
team_label.pack(pady=(15, 5))

support_label = ctk.CTkLabel(footer, text=f"Support: {SUPPORT_EMAIL}", font=("Segoe UI", 12), text_color="#94A3B8")
support_label.pack()

feedback_button = ctk.CTkButton(
    footer, text="Contact / Feedback", command=open_feedback_window,
    width=220, height=42, font=("Segoe UI", 13, "bold"), corner_radius=12,
    fg_color="#0EA5E9", hover_color="#0284C7"
)
feedback_button.pack(pady=(10, 20))


main_frame = ctk.CTkFrame(app, fg_color="#0F172A", corner_radius=0)
main_frame.pack(side="top", fill="both", expand=True)

card = ctk.CTkFrame(main_frame, fg_color="#1E293B", corner_radius=25, border_width=1, border_color="#334155")
card.pack(expand=True, padx=40, pady=40)

dashboard_title = ctk.CTkLabel(card, text="Signal Analysis Dashboard", font=("Segoe UI", 28, "bold"))
dashboard_title.pack(pady=(35, 10))

dashboard_desc = ctk.CTkLabel(card, text="Upload a radar signal CSV file with 5000 features", font=("Segoe UI", 14), text_color="#CBD5E1")
dashboard_desc.pack(pady=(0, 25))

upload_button = ctk.CTkButton(
    card, text="Upload Signal File", command=classify_signal,
    width=250, height=55, font=("Segoe UI", 16, "bold"), corner_radius=14,
    fg_color="#2563EB", hover_color="#1D4ED8"
)
upload_button.pack()

icon_label = ctk.CTkLabel(card, text="📡", font=("Segoe UI Emoji", 70))
icon_label.pack(pady=(25, 5))

result_label = ctk.CTkLabel(card, text="Awaiting Signal Input", font=("Segoe UI", 32, "bold"), text_color="white")
result_label.pack(pady=5)

status_label = ctk.CTkLabel(card, text="System Ready", font=("Segoe UI", 14), text_color="#94A3B8")
status_label.pack(pady=(0, 25))

stats_frame = ctk.CTkFrame(card, fg_color="#111827", corner_radius=18, width=780)
stats_frame.pack(pady=(0, 30), padx=40)

accuracy = ctk.CTkLabel(stats_frame, text="Accuracy\n98.5%", font=("Segoe UI", 18, "bold"), text_color="#10B981")
accuracy.pack(side="left", expand=True, pady=20, padx=20)

algorithm = ctk.CTkLabel(stats_frame, text="Algorithm\nSVM", font=("Segoe UI", 18, "bold"), text_color="#3B82F6")
algorithm.pack(side="left", expand=True, pady=20, padx=20)

deployment = ctk.CTkLabel(stats_frame, text="Deployment\nReal-Time", font=("Segoe UI", 18, "bold"), text_color="#F59E0B")
deployment.pack(side="left", expand=True, pady=20, padx=20)

app.mainloop()