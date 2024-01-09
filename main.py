import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import numpy as np
import cv2
from PIL import Image, ImageTk
import face_recognition
import os
import time
from datetime import datetime

iou_threshold = 0.5

class OpeningPage:
    def __init__(self, root, on_start_callback):
        self.root = root
        self.root.title("Opening Page")
        self.root.geometry("600x400")

        # Create a label and button on the opening page
        label = tk.Label(root, text="Click the button to start the session")
        label.pack(pady=20)

        start_button = tk.Button(root, text="Start Session", command=on_start_callback)
        start_button.pack(pady=10)

        new_student_button = tk.Button(root, text="Add New Student", command=self.open_new_student_page)
        new_student_button.pack(pady=10)

    def open_new_student_page(self):
        # Open the New Student page
        new_student_page = NewStudentPage(self.root)
        new_student_page.run()

    def run(self):
        self.root.mainloop()

class StudentApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Student App")

        # Add "Attendance Status" and "Attendance Time" columns to the students DataFrame
        self.students = pd.DataFrame(columns=["StudentID", "Fullname", "Attendance Status", "Attendance Time"])

        # Create and populate the student listbox
        self.student_listbox = tk.Listbox(root, selectmode=tk.SINGLE, width=60)  # Adjust width as needed
        self.student_listbox.grid(row=1, column=0, padx=10, pady=10, rowspan=5)
        self.populate_student_listbox()

        # Create import button
        import_button = tk.Button(root, text="Import Students", command=self.import_students)
        import_button.grid(row=0, column=0, padx=10, pady=10)

        # Create camera view
        self.camera_label = tk.Label(root)
        self.camera_label.grid(row=1, column=1, padx=10, pady=10, rowspan=5)
        self.capture_camera()

        # Example images for face recognition
        self.faces = {}
        self.load_faces()

        # Create End Session button
        end_session_button = tk.Button(root, text="End Session", command=self.end_session)
        end_session_button.grid(row=6, column=0, padx=10, pady=10)

    def populate_student_listbox(self):
        self.student_listbox.delete(0, tk.END)
        for index, row in self.students.iterrows():
            attendance_status = row["Attendance Status"] if pd.notnull(row["Attendance Status"]) else ""
            self.student_listbox.insert(tk.END, f"{row['StudentID']} - {row['Fullname']} - {attendance_status}")

    def import_students(self):
        file_path = filedialog.askopenfilename(title="Select Excel File", filetypes=[("Excel files", "*.xlsx;*.xls")])
        if file_path:
            try:
                data = pd.read_excel(file_path)
                data['StudentID'] = data['StudentID'].apply(lambda x: str(x))
                # Add "Attendance Status" and "Attendance Time" columns to the imported students DataFrame
                data["Attendance Status"] = ""
                data["Attendance Time"] = ""
                print(data)
                self.students = data
                self.populate_student_listbox()
            except Exception as e:
                print(f"Error reading Excel file: {e}")

    def calculate_iou(self, rect1, rect2):
        # Calculate IoU between two rectangles
        x1, y1, w1, h1 = rect1
        x2, y2, w2, h2 = rect2

        # Intersection rectangle
        x_intersect = max(x1, x2)
        y_intersect = max(y1, y2)
        w_intersect = min(x1 + w1, x2 + w2) - x_intersect
        h_intersect = min(y1 + h1, y2 + h2) - y_intersect

        if w_intersect > 0 and h_intersect > 0:
            area_intersect = w_intersect * h_intersect
            area_rect1 = w1 * h1
            area_rect2 = w2 * h2
            iou = area_intersect / (area_rect1 + area_rect2 - area_intersect)
            return iou
        else:
            return 0

    def load_faces(self):
        # Load example faces for face recognition from all PNG images in the faces folder
        faces_folder = "faces"
        self.faces = {}

        for filename in os.listdir(faces_folder):
            if filename.endswith(".png"):
                example_face_path = os.path.join(faces_folder, filename)
                example_face = face_recognition.load_image_file(example_face_path)
                example_face_encoding = face_recognition.face_encodings(example_face)[0]
                self.faces[filename.split('_')[0]] = example_face_encoding

    def recognize_face(self, face_image):
        # Recognize face using face recognition library
        unknown_face_encoding = face_recognition.face_encodings(face_image)

        if not unknown_face_encoding:
            return None

        unknown_face_encoding = unknown_face_encoding[0]

        ids = list(self.faces.keys())

        faces = []
        for key in ids:
            faces.append(self.faces[key])

        # Compare the face with example faces
        matches = face_recognition.compare_faces(faces, unknown_face_encoding)
        face_distances = face_recognition.face_distance(faces, unknown_face_encoding)

        # Find the most similar face
        best_match_index = np.argmin(face_distances)

        if matches[best_match_index]:
            return ids[best_match_index]  # Return StudentID if matched
        else:
            return None

    def capture_camera(self):
        cap = cv2.VideoCapture(1)  # Use 0 for default camera

        # Load Haarcascades for face detection
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        def update_camera():
            ret, frame = cap.read()
            if ret:
                # Detect faces
                faces = face_cascade.detectMultiScale(frame, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

                # Draw rectangles around the faces
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

                    # Draw rectangle at the center for face placement guide
                    center_rectangle_size = (200, 220)
                    center_rectangle_x = (frame.shape[1] - center_rectangle_size[0]) // 2
                    center_rectangle_y = (frame.shape[0] - center_rectangle_size[1]) // 2
                    cv2.rectangle(frame, (center_rectangle_x, center_rectangle_y),
                                  (center_rectangle_x + center_rectangle_size[0], center_rectangle_y + center_rectangle_size[1]),
                                  (0, 255, 0), 2)

                    # Calculate IoU between the face and the center rectangle
                    iou = self.calculate_iou((x, y, w, h), (center_rectangle_x, center_rectangle_y, center_rectangle_size[0], center_rectangle_size[1]))

                    # Show a pop-up if the face fits well within the center rectangle
                    if iou > iou_threshold:
                        # Recognize the face and update the attendance status
                        result = self.recognize_face(frame)
                        if result is not None:
                            indexes = self.students[self.students["StudentID"] == result].index.values
                            if len(indexes) > 0:
                                index = indexes[0]
                                if self.students.at[index, 'Attendance Status'] != "":
                                    messagebox.showinfo("Error",
                                                        f"{self.students.at[index, 'Fullname']} already placed in the attendance list.")
                                else:
                                    self.students.at[index, 'Attendance Status'] = "Present"
                                    self.students.at[index, 'Attendance Time'] = time.strftime("%Y-%m-%d %H:%M:%S")

                                    print(self.students)
                                    # Update the listbox to reflect changes
                                    self.populate_student_listbox()

                                    messagebox.showinfo("Face Placement",
                                                        f"Face is perfectly placed!\nRecognized as: {result}")
                            else:
                                messagebox.showinfo("Student not found",
                                                    f"Student with {result} id is not registered for this course.")
                        else:
                            messagebox.showinfo("No Match",
                                                "There is no face that matches with this.")

                # Display the image
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame)
                img = ImageTk.PhotoImage(img)
                self.camera_label.img = img
                self.camera_label.config(image=img)
                self.root.after(10, update_camera)

        update_camera()

    def end_session(self):
        # Export students DataFrame to an Excel file in the "outputs" folder
        output_folder = "outputs"
        os.makedirs(output_folder, exist_ok=True)
        output_file_path = os.path.join(output_folder, f"attendance_{datetime.now().strftime('%Y%m%d%H%M%S')}.xlsx")
        self.students.to_excel(output_file_path, index=False)
        messagebox.showinfo("End Session", f"Attendance exported to {output_file_path}")
        self.root.destroy()

    def run(self):
        self.root.mainloop()

class NewStudentPage:
    def __init__(self, master):
        self.master = master
        self.root = tk.Toplevel(master)
        self.root.title("New Student Page")
        self.root.geometry("600x800")

        # Create and initialize variables for StudentID, Fullname, and captured face
        self.student_id_var = tk.StringVar()
        self.fullname_var = tk.StringVar()
        self.captured_face = None

        # Create labels and entry widgets for StudentID and Fullname
        tk.Label(self.root, text="StudentID:").pack(pady=10)
        tk.Entry(self.root, textvariable=self.student_id_var).pack(pady=10)
        tk.Label(self.root, text="Fullname:").pack(pady=10)
        tk.Entry(self.root, textvariable=self.fullname_var).pack(pady=10)

        # Create label for displaying video capture preview
        self.preview_label = tk.Label(self.root)
        self.preview_label.pack(pady=10)

        # Button to start/stop capturing face
        self.capture_button = tk.Button(self.root, text="Capture Face", command=self.capture_face)
        self.capture_button.pack(pady=10)

        self.capture_button = tk.Button(self.root, text="Save", command=self.save_student)
        self.capture_button.pack(pady=10)

        # Initialize video capture
        self.cap = cv2.VideoCapture(1)  # Use 0 for default camera
        self.update_preview()

    def capture_face(self):
        # Capture face and update the captured_face variable
        ret, frame = self.cap.read()
        if ret:
            self.captured_face = frame
            messagebox.showinfo("Capture Face", "Face captured successfully!")

    def update_preview(self):
        # Update the video capture preview
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            img = ImageTk.PhotoImage(img)
            self.preview_label.img = img
            self.preview_label.config(image=img)

        self.root.after(10, self.update_preview)

    def save_student(self):
        # Save the new student to the dataframe and save the captured face to a file
        student_id = self.student_id_var.get()
        fullname = self.fullname_var.get()

        if student_id and fullname and self.captured_face is not None:
            # Save the captured face image to a file
            faces_folder = "faces"
            os.makedirs(faces_folder, exist_ok=True)
            face_image_path = os.path.join(faces_folder, f"{student_id}_{fullname}.png")
            cv2.imwrite(face_image_path, self.captured_face)

            # Add the new student to the students DataFrame
            new_student = pd.DataFrame({"StudentID": [student_id],
                                         "Fullname": [fullname],
                                         "Attendance Status": [""],
                                         "Attendance Time": [""]})
            # self.master.students = pd.concat([self.master.students, new_student], ignore_index=True)

            messagebox.showinfo("Save Student", "Student and face saved successfully!")
            self.root.destroy()
        else:
            messagebox.showerror("Save Student", "Please enter StudentID, Fullname, and capture a face.")

    def run(self):
        self.root.after(1, self.update_preview)  # Start updating the preview
        self.root.mainloop()

def main():
    def start_session():
        opening_page.root.destroy()  # Close the opening page
        root = tk.Tk()
        app = StudentApp(root)
        app.run()

    opening_root = tk.Tk()
    opening_page = OpeningPage(opening_root, start_session)
    opening_page.run()

if __name__ == "__main__":
    main()
