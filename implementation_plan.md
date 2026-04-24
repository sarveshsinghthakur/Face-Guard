# Facial Recognition Attendance System Implementation Plan

This plan details the implementation of the new features you requested: managing a persistent user registry, dynamically registering new faces via UI prompts, sending email notifications upon attendance, and tracking daily attendance to prevent duplicate emails.

## Proposed Changes

### 1. Refactoring `mail.py`
We will wrap the existing email logic in `mail.py` into a reusable function that takes a recipient email and their name.

#### [MODIFY] `mail.py`
- Create a `send_attendance_mail(receiver_email, name)` function.
- Dynamically inject the user's name and email into the subject/body instead of using the hardcoded values.
- Handle any potential connection errors gracefully without crashing the main application.

### 2. User Registry (`users.csv`)
We will introduce a central `users.csv` file that maps a user's Name to their Email address. 

#### [MODIFY] `face_recignition.py`
- Create helper functions to load and save to `users.csv`.
- On startup, read `users.csv` into a dictionary: `user_emails = {"Name": "email@example.com"}`.

### 3. Attendance Tracking Logic
Currently, the script uses a `students = known_names.copy()` list and removes items to track attendance. This doesn't work well for dynamically added users.

#### [MODIFY] `face_recignition.py`
- Replace the `students` list with a `recorded_today = set()` which keeps track of who has been processed.
- When a known face is recognized:
  - Check if `name in recorded_today`.
  - If yes, ignore (they are already marked).
  - If no, add them to `recorded_today`, record their attendance in `dd-mm-yyyy.csv`, and invoke `send_attendance_mail()`.

### 4. Handling Unknown Faces (New Registrations)
When the system detects a face it doesn't recognize (i.e., `match_face()` returns `"Unknown"`), it will prompt for registration.

#### [MODIFY] `face_recignition.py`
- To avoid freezing the camera feed or switching context to the terminal, we will use Python's built-in `tkinter` library to show a small popup dialog asking for the **Name** and **Email** of the new person.
- Once provided:
  - We will save the cropped face image directly into the `known_faces/` folder as `<Name>.jpg`.
  - Add their details to `users.csv`.
  - Re-compute their encoding and add it to the active `known_encodings` list.
  - Automatically mark them as present, log them to today's CSV, and send them a welcome/attendance email.
  - *Debounce mechanism*: We'll add a temporary "pause" on unknown faces right after a registration prompt is closed to prevent the dialog from spamming if the user cancels.

## Open Questions

> [!NOTE]
> 1. **Popup Dialog**: I am planning to use a `tkinter` dialog popup to ask for the Name and Email because terminal inputs can freeze the OpenCV webcam window and are hard to notice. Does this sound good to you?
> 2. **Camera Pause**: When the popup appears, the video feed will pause until you submit the details or hit cancel. Is this acceptable?
> 3. **Passwords**: `mail.py` currently contains your app password. It will remain exactly as is, but just wrapped in a function.

Please review this plan. If you approve, I will proceed with the code changes.
