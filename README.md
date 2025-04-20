

---

# 🧍‍♂️ Gait Analysis Using MediaPipe

This project performs **gait analysis** by extracting and analyzing **lower limb joint angles** from a video of a walking subject. It uses **MediaPipe's Pose Estimation**, **OpenCV**, and **Matplotlib** to track and visualize the motion of the **hips, knees, and ankles** throughout the gait cycle.

Developed a comprehensive **gait analysis tool** leveraging MediaPipe’s pose estimation and signal processing techniques to calculate and visualize joint angles of the lower body during walking. The system analyzes **knee, hip, and ankle dynamics**, identifies **gait cycles**, and provides **comparative visualizations** of left vs. right limb behavior. Additional features include **angle smoothing** using the **Savitzky-Golay filter** and **animated motion visualization** to enhance interpretability.

---

## 📹 Demo

The system takes a **walking video** as input and outputs:

- 📈 **Angle plots** of left and right **knee, hip, and ankle**
- 🔄 **Gait cycle graph** showing angle changes across a step
- 🎞️ **Animated visualization** of limb angles over time

---

## 🧰 Libraries Used

- **OpenCV** – Video processing  
- **MediaPipe** – Pose landmark detection  
- **NumPy** – Vector math and angle calculations  
- **SciPy** – Smoothing and peak detection  
- **Matplotlib** – Plotting and animations  

---

## ⚙️ How It Works

1. **Pose Estimation**  
   MediaPipe detects body landmarks for each frame.

2. **Joint Angle Calculation**  
   Angles at hip, knee, and ankle are calculated using 2D vector math.

3. **Data Smoothing**  
   Savitzky-Golay filter is applied to reduce noise in joint angle data.

4. **Gait Cycle Detection**  
   Knee angles are analyzed to segment one full gait cycle.

5. **Visualization**  
   Time-series plots, gait cycle graphs, and animated plots are generated for analysis.

---

## 📊 Outputs

### ✅ Angle Comparison
Plots of left vs right angles for:
- **Knee**
- **Ankle**
- **Hip**

### ✅ Gait Cycle Plot
- A **normalized plot** showing joint angles across one gait cycle.

### ✅ Animated Graph *(Optional)*
- **Live animation** showing how angles change over time.

---

## 📌 Notes

- Ensure the input video provides a **clear side view** of walking for accurate joint detection.
- Accuracy and performance depend on **video quality** and **pose detection confidence**.
- Adjust **smoothing `window_size`** if needed for better angle stability.

---

## 🏥 Applications

- Medical **gait analysis**
- **Sports** performance tracking
- **Rehabilitation** monitoring
- **Biomechanics** education and research

---

