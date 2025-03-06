# ###################################################################################
# Software Name: ddLFIA Voting Model Weighting Algorithm Detection Tool
# Function Overview:
# This software develops the ddLFIA Voting Model Weighting Algorithm (VMWA) using the Ant Colony Optimization (ACO) algorithm.
# It combines image processing and machine learning techniques to optimize the parameters of adaptive threshold segmentation,
# and combines with the DBSCAN clustering algorithm to achieve image region segmentation, improving the accuracy and robustness of image analysis.
# Specifically, it uses the photothermal detection algorithm to extract image features, and respectively uses the Linear Regression and Gradient Boosting Regression (GBR) models
# to quantitatively predict the concentrations of BaP and DBP.
# Then, a weighted voting mechanism based on the RMSE and R2 of the two models is used to fuse the prediction results to achieve efficient and accurate quantification of BaP and DBP.
# Finally, a Graphical User Interface (GUI) is created through Python and Tkinter to facilitate user operation.
# Users can select or open images and manually input temperature values.
# The software will automatically process the images and predict the concentrations of BaP and DBP according to the specified limits, and display the corresponding warning information.
#
# Code Module Description:
# 1. ModelBuilder class: Responsible for building Linear Regression and Gradient Boosting Regression models,
#    which are used to predict the concentrations of BaP and DBP, and calculate the evaluation metrics (MSE and R²) of the models.
# 2. AntColonyOptimizer class: Implements the Ant Colony Optimization algorithm to find the optimal parameters (block_size and C value) for adaptive threshold segmentation.
# 3. ImageProcessor class: Uses the parameters obtained by Ant Colony Optimization to process images,
#    including threshold segmentation, contour detection, DBSCAN clustering, etc., to extract feature gray values.
# 4. WeightedVoting class: Provides a static method for weighted voting prediction to fuse the prediction results of different models.
# 5. ImageProcessorApp class: Creates a GUI interface to handle user interactions, such as selecting images, inputting temperatures,
#    displaying results and warning information, etc.
#
# Notes:
# - Make sure that the required Python libraries (such as numpy, opencv, scikit-learn, pandas, Pillow, tkinter, etc.) are correctly installed before running the code.
# ###################################################################################

import warnings
import numpy as np
import cv2
import tkinter as tk
import time
from tkinter import filedialog, messagebox
from tkinter import font as tkfont
from PIL import Image, ImageTk, ImageGrab
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor

# Suppress warning messages
warnings.filterwarnings("ignore")


class ModelBuilder:
    def __init__(self):
        # Initialize the relevant data of BaP and DBP
        self.BaP = np.array([0.05, 0.5, 1, 5, 10, 50, 100, 300])
        self.gray_bap = np.array([152.27, 157.21, 166, 175.48, 179.01, 188.1, 192.24, 196.47])
        self.DBP = np.array([0.5, 1, 5, 20, 100, 300, 500, 1000])
        self.gray_dbp = np.array([152, 154.62, 165.44, 176.11, 182.1, 189.14, 192.78, 197.33])

        # Build the models
        self.build_bap_models()
        self.build_dbp_models()

    def build_bap_models(self):
        log_BaP = np.log(self.BaP)
        self.model_bap = LinearRegression()
        self.model_bap.fit(self.gray_bap.reshape(-1, 1), log_BaP)

        # Model 1 for BaP
        data_bap = {
            '浓度': [0.05, 0.5, 1, 5, 10, 50, 100, 300,
                     0.05, 0.5, 1, 5, 10, 50, 100, 300],
            'T': [21.4, 18.2, 16.4, 13.1, 11.3, 9.2, 6.8, 3.1,
                  20.7, 17.4, 15.5, 13.8, 11.9, 8.5, 6.2, 2.4]
        }
        df_bap = pd.DataFrame(data_bap)
        X_bap = df_bap[['T']]
        y_bap = df_bap['浓度']
        X_train_bap, X_test_bap, y_train_bap, y_test_bap = train_test_split(X_bap, y_bap, test_size=0.2,
                                                                            random_state=42)

        self.model_bap_all = RandomForestRegressor(random_state=42)
        self.model_bap_all.fit(X_train_bap, y_train_bap)

        bap_light_pred = self.model_bap.predict(self.gray_bap.reshape(-1, 1))
        bap_img_pred = self.model_bap_all.predict(X_test_bap)
        self.mse_light_bap = mean_squared_error(log_BaP, bap_light_pred)
        self.mse_img_bap = mean_squared_error(y_test_bap, bap_img_pred)
        self.r2_light_bap = r2_score(log_BaP, bap_light_pred)
        self.r2_img_bap = r2_score(y_test_bap, bap_img_pred)

    def build_dbp_models(self):
        log_DBP = np.log(self.DBP)
        self.model_dbp = LinearRegression()
        self.model_dbp.fit(self.gray_dbp.reshape(-1, 1), log_DBP)

        # Model 2 for DBP
        data_dbp = {
            '浓度': [0.5, 1, 5, 20, 100, 300, 500, 1000,
                     0.5, 1, 5, 20, 100, 300, 500, 1000],
            'T': [23.4, 20.2, 16.4, 14.1, 11.3, 9.2, 6.7, 4.9,
                  24.2, 21.1, 17.5, 13.4, 11.9, 8.5, 6.1, 4.2]
        }
        df_dbp = pd.DataFrame(data_dbp)
        X_dbp = df_dbp[['T']]
        y_dbp = df_dbp['浓度']
        X_train_dbp, X_test_dbp, y_train_dbp, y_test_dbp = train_test_split(X_dbp, y_dbp, test_size=0.2,
                                                                            random_state=42)

        self.model_dbp_all = RandomForestRegressor(random_state=42)
        self.model_dbp_all.fit(X_train_dbp, y_train_dbp)

        dbp_light_pred = self.model_dbp_all.predict(X_test_dbp)
        dbp_img_pred = self.model_dbp.predict(self.gray_dbp.reshape(-1, 1))
        self.mse_light_dbp = mean_squared_error(y_test_dbp, dbp_light_pred)
        self.mse_img_dbp = mean_squared_error(log_DBP, dbp_img_pred)
        self.r2_light_dbp = r2_score(y_test_dbp, dbp_light_pred)
        self.r2_img_dbp = r2_score(log_DBP, dbp_img_pred)


class AntColonyOptimizer:
    def __init__(self):
        # Parameter settings for the Ant Colony algorithm
        self.num_ants = 10  # Number of ants
        self.num_iterations = 50  # Number of iterations
        self.alpha = 1  # Importance of pheromone
        self.beta = 2  # Importance of the heuristic function
        self.rho = 0.1  # Evaporation rate of pheromone
        self.Q = 100  # Constant for pheromone increment
        self.block_sizes = [3, 5, 7, 9, 11]  # Optional block_size
        self.C_values = [1, 2, 3, 4, 5]  # Optional C values

    def evaluate_threshold_params(self, image, block_size, C):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        binary = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            block_size,
            C
        )
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return 0
        max_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(max_contour)
        area = w * h
        return area  # 使用轮廓的面积作为分割效果的评估标准

    def initialize_pheromone_matrix(self):
        pheromone_matrix = np.ones((len(self.block_sizes), len(self.C_values)))
        return pheromone_matrix

    def select_param(self, pheromone_matrix, alpha, beta, length):
        pheromone_values = pheromone_matrix[:, 0]
        probability = pheromone_values ** alpha
        total_pheromone = sum(probability)
        probability /= total_pheromone
        return np.random.choice(range(length), p=probability)

    def update_pheromone(self, pheromone_matrix, all_ants_solutions):
        pheromone_matrix *= (1 - self.rho)
        for score, block_size, C, block_size_index, C_index in all_ants_solutions:
            pheromone_matrix[block_size_index, C_index] += self.Q / (1 + score)

    def ant_colony_optimization(self, image):
        pheromone_matrix = self.initialize_pheromone_matrix()
        best_score = 0
        best_block_size = 0
        best_C = 0
        for iteration in range(self.num_iterations):
            all_ants_solutions = []
            for ant in range(self.num_ants):
                block_size_index = self.select_param(pheromone_matrix, self.alpha, self.beta, len(self.block_sizes))
                C_index = self.select_param(pheromone_matrix, self.alpha, self.beta, len(self.C_values))
                block_size = self.block_sizes[block_size_index]
                C = self.C_values[C_index]
                score = self.evaluate_threshold_params(image, block_size, C)
                all_ants_solutions.append((score, block_size, C, block_size_index, C_index))
                if score > best_score:
                    best_score = score
                    best_block_size = block_size
                    best_C = C
            self.update_pheromone(pheromone_matrix, all_ants_solutions)
        return best_block_size, best_C


class ImageProcessor:
    def __init__(self):
        self.aco = AntColonyOptimizer()

    def process_image_dbscan(self, image_path, output_path):
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("Failed to read the image. Please check the path and file format.")
        best_block_size, best_C = self.aco.ant_colony_optimization(image)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        binary = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            best_block_size,
            best_C
        )
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            raise ValueError("No contours found. Please check the image content or adjust the threshold.")
        max_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(max_contour)
        roi = image[y:y + h, :]
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        grad_x = cv2.Sobel(gray_roi, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray_roi, cv2.CV_64F, 0, 1, ksize=3)
        grad_magnitude = cv2.magnitude(grad_x, grad_y)

        data = grad_magnitude.flatten().reshape(-1, 1)  # 展平为一维数组用于聚类
        dbscan = DBSCAN(eps=0.1, min_samples=5)  # 根据需求调整eps和min_samples
        labels = dbscan.fit_predict(data)

        unique_labels = np.unique(labels)
        boundaries = []
        region_areas = []
        region_means = []

        for label in unique_labels:
            if label != -1:
                region_indices = np.where(labels == label)[0]
                start = region_indices[0]
                end = region_indices[-1]

                region_image = gray_roi[:, start:end]
                area = region_image.size
                region_mean = np.nanmean(region_image) if not np.isnan(np.nanmean(region_image)) else 0
                region_areas.append(area)
                region_means.append(region_mean)
                boundaries.append((start, end))
        max_area_index = np.argmax(region_areas)
        image_mean_gray_value = region_means[max_area_index]
        for start, end in boundaries:
            cv2.rectangle(roi, (start, 0), (end, roi.shape[0]), (255, 0, 0), 2)
        roi = cv2.resize(roi, (100, 30))
        return gray_roi, roi, image_mean_gray_value


class WeightedVoting:
    @staticmethod
    def weighted_voting_prediction(*args):
        """
        Calculate the weighted average prediction value.
        Parameters: The predicted values and weights appear alternately.
        :param args: Parameters with predicted values and weights appearing alternately.
        :return: The final predicted value.
        """
        if len(args) % 2 != 0:
            raise ValueError("The number of predicted values and weights must appear in pairs.")
        predictions = args[0::2]  # Get all the predicted values
        weights = args[1::2]  # Get all the weights
        if len(predictions) == 0 or len(weights) == 0:
            raise ValueError("No valid predicted values or weights were provided.")
        # Print the predicted values and weights
        print(f"Predictions: {predictions}, Weights: {weights}")
        # Calculate the weighted average
        weighted_sum = sum(p * w for p, w in zip(predictions, weights))
        return weighted_sum / sum(weights)


class ImageProcessorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("ddLFIA Voting Model Weighting Algorithm")
        self.root.configure(bg='#f0f0f0')  # Set background color
        # Initialize the model builder
        self.model_builder = ModelBuilder()
        # Initialize the image processor
        self.image_processor = ImageProcessor()
        # Store the average gray values of BaP and DBP
        self.mean_values = {}
        # Store the prediction results of BaP and DBP
        self.result = {}
        # Store the final prediction results of BaP and DBP
        self.forecast = {}
        # Store the labels of original images
        self.left_frames = []
        # Store the labels of processed images
        self.right_frames = []
        # Store the labels of average gray values
        self.mean_labels = []
        # Create the GUI interface
        self.create_widgets()

    def create_widgets(self):
        for i in range(2):
            left_frame = tk.Frame(self.root, padx=10, pady=10, bg='#e0e0e0', relief=tk.RAISED, borderwidth=2)
            left_frame.grid(row=i, column=0, padx=10, pady=10)
            left_label = tk.Label(left_frame, text=f"Original Image {i + 1}", bg='#e0e0e0',
                                  font=('Helvetica', 12, 'bold'))
            left_label.pack()
            self.left_frames.append(left_label)

            right_frame = tk.Frame(self.root, padx=10, pady=10, bg='#e0e0e0', relief=tk.RAISED, borderwidth=2)
            right_frame.grid(row=i, column=1, padx=10, pady=10)
            right_label = tk.Label(right_frame, text=f"Processed Image {i + 1}", bg='#e0e0e0',
                                   font=('Helvetica', 12, 'bold'))
            right_label.pack()
            self.right_frames.append(right_label)

            mean_value_label = tk.Label(self.root, text="Feature grey", bg='#f0f0f0', font=('Helvetica', 12, 'bold'))
            mean_value_label.grid(row=i, column=2, padx=10, pady=10)
            mean_label = tk.Label(mean_value_label, text="Feature grey: ", bg='#f0f0f0', font=('Helvetica', 12))
            mean_label.pack()
            self.mean_labels.append(mean_label)

        select_BaP_button = tk.Button(self.root, text="Select BaP Image",
                                      command=lambda: self.capture_screen(0, 'BaP'),
                                      bg='#4CAF50', fg='white', font=('Helvetica', 12, 'bold'),
                                      relief=tk.RAISED, borderwidth=2)
        select_BaP_button.grid(row=3, column=0, padx=10, pady=10)

        select_DBP_button = tk.Button(self.root, text="Select DBP Image",
                                      command=lambda: self.capture_screen(1, 'DBP'),
                                      bg='#2196F3', fg='white', font=('Helvetica', 12, 'bold'),
                                      relief=tk.RAISED, borderwidth=2)
        select_DBP_button.grid(row=3, column=1, padx=10, pady=10)

        open_BaP_button = tk.Button(self.root, text="Open BaP Image",
                                    command=lambda: self.open_file(0, 'BaP'),
                                    bg='#FF5722', fg='white', font=('Helvetica', 12, 'bold'),
                                    relief=tk.RAISED, borderwidth=2)
        open_BaP_button.grid(row=4, column=0, padx=10, pady=10)

        open_DBP_button = tk.Button(self.root, text="Open DBP Image",
                                    command=lambda: self.open_file(1, 'DBP'),
                                    bg='#3F51B5', fg='white', font=('Helvetica', 12, 'bold'),
                                    relief=tk.RAISED, borderwidth=2)
        open_DBP_button.grid(row=4, column=1, padx=10, pady=10)

        reset_button = tk.Button(self.root, text="Reset", command=self.reset,
                                 bg='#f44336', fg='white', font=('Helvetica', 12, 'bold'),
                                 relief=tk.RAISED, borderwidth=2)
        reset_button.grid(row=3, column=2, padx=10, pady=10)

        calc_BaP_button = tk.Button(self.root, text="Calculate BaP", command=self.calculate_bap,
                                    bg='#FF9800', fg='white', font=('Helvetica', 12, 'bold'),
                                    relief=tk.RAISED, borderwidth=2)
        calc_BaP_button.grid(row=3, column=3, padx=10, pady=10)

        calc_DBP_button = tk.Button(self.root, text="Calculate DBP", command=self.calculate_dbp,
                                    bg='#9C27B0', fg='white', font=('Helvetica', 12, 'bold'),
                                    relief=tk.RAISED, borderwidth=2)
        calc_DBP_button.grid(row=4, column=3, padx=10, pady=10)

        self.bg_color = '#f0f0f0'
        self.entry_bg = '#ffffff'
        self.button_bg = '#4CAF50'
        self.button_fg = '#ffffff'
        self.button_hover_bg = '#45a049'
        self.entry_border_color = '#d0d0d0'

        self.custom_font = tkfont.Font(family='Helvetica', size=12)
        self.bold_font = tkfont.Font(family='Helvetica', size=12, weight='bold')

        self.create_label("Bap_T（℃）:", 5, 0)
        self.create_label("DBP_T（℃）:", 6, 0)

        self.bap_t_var = tk.DoubleVar()
        self.dbp_t_var = tk.DoubleVar()

        self.create_entry(self.bap_t_var, 5, 1)
        self.create_entry(self.dbp_t_var, 6, 1)

        self.create_button("Calculate Bap all", self.calculate_bap_all, 5, 2)
        self.create_button("Calculate DBP all", self.calculate_dbp_all, 6, 2)

        try:
            image_path = "config/Background_image.png"
            original_image = Image.open(image_path)
            resized_image = original_image.resize((110, 80))
            self.image = ImageTk.PhotoImage(resized_image)
            image_label = tk.Label(self.root, image=self.image)
            image_label.grid(row=5, column=3, rowspan=2, columnspan=2, padx=10, pady=10)

        except Exception as e:
            print(f"Error loading image: {e}")

        self.bap_forecast_result_label = tk.Label(self.root, text="Please calculate Bap.",
                                                  bg=self.bg_color, font=self.bold_font)
        self.bap_forecast_result_label.grid(row=8, column=0, columnspan=2, padx=10, pady=10, sticky='w')
        self.bap_warning_label = tk.Label(self.root, text="", bg=self.bg_color, font=self.bold_font)
        self.bap_warning_label.grid(row=8, column=2, padx=0, pady=0, sticky='w')
        self.dbp_forecast_result_label = tk.Label(self.root, text="Please calculate DBP.",
                                                  bg=self.bg_color, font=self.bold_font)
        self.dbp_forecast_result_label.grid(row=9, column=0, columnspan=2, padx=10, pady=10, sticky='w')
        self.dbp_warning_label = tk.Label(self.root, text="", bg=self.bg_color, font=self.bold_font)
        self.dbp_warning_label.grid(row=9, column=2, padx=0, pady=0, sticky='w')

    def create_label(self, text, row, column):
        label = tk.Label(self.root, text=text, bg=self.bg_color, font=self.custom_font)
        label.grid(row=row, column=column, padx=0, pady=0, sticky='e')

    def create_entry(self, variable, row, column):
        entry = tk.Entry(self.root, textvariable=variable, bg=self.entry_bg, font=self.custom_font,
                         borderwidth=2, relief='solid')
        entry.grid(row=row, column=column, padx=0, pady=0, sticky='w')
        entry.config(highlightbackground=self.entry_border_color, highlightcolor=self.entry_border_color,
                     highlightthickness=1)

    def create_button(self, text, command, row, column):
        button = tk.Button(self.root, text=text, command=command, bg=self.button_bg, fg=self.button_fg,
                           font=self.custom_font, relief='raised', bd=2)
        button.grid(row=row, column=column, padx=10, pady=10)
        button.bind("<Enter>", lambda e: button.config(bg=self.button_hover_bg))
        button.bind("<Leave>", lambda e: button.config(bg=self.button_bg))

    def open_file(self, index, category):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp")])
        if file_path:
            self.process_and_display_image(file_path, index, category)

    def process_and_display_image(self, file_path, index, category):
        try:
            original_image = Image.open(file_path)
            original_image = original_image.resize((100, 30))

            gray_roi, processed_image, mean_processed = self.image_processor.process_image_dbscan(file_path,
                                                                                                  'temp_processed_image.png')
            original_image_tk = ImageTk.PhotoImage(original_image)
            processed_image_tk = ImageTk.PhotoImage(Image.fromarray(processed_image))
            self.left_frames[index].config(image=original_image_tk)
            self.left_frames[index].image = original_image_tk
            self.right_frames[index].config(image=processed_image_tk)
            self.right_frames[index].image = processed_image_tk
            self.mean_labels[index].config(text=f"Feature grey: {round(mean_processed, 2)}")
            self.mean_values[category] = mean_processed
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def capture_screen(self, index, label):
        self.root.withdraw()
        screenshot = ImageGrab.grab()
        screenshot_path = f"screenshot_{index + 1}.png"
        screenshot.save(screenshot_path)
        selector = tk.Toplevel(self.root)
        selector.geometry(f"{screenshot.width}x{screenshot.height}+0+0")
        selector.attributes("-fullscreen", True)
        selector.attributes("-topmost", True)

        screenshot_tk = ImageTk.PhotoImage(screenshot)
        canvas = tk.Canvas(selector, width=screenshot.width, height=screenshot.height)
        canvas.create_image(0, 0, anchor=tk.NW, image=screenshot_tk)
        canvas.pack()

        start_x = start_y = end_x = end_y = 0

        def on_mouse_down(event):
            nonlocal start_x, start_y
            start_x, start_y = event.x, event.y
            canvas.delete("rect")

        def on_mouse_drag(event):
            nonlocal end_x, end_y
            end_x, end_y = event.x, event.y
            canvas.delete("rect")
            canvas.create_rectangle(start_x, start_y, end_x, end_y, outline="red", tag="rect")

        def on_mouse_up(event):
            nonlocal end_x, end_y
            end_x, end_y = event.x, event.y
            selector.destroy()

            bbox = (min(start_x, end_x), min(start_y, end_y), max(start_x, end_x), max(start_y, end_y))
            selected_region = screenshot.crop(bbox)
            selected_region_path = f"selected_region_{index + 1}.png"
            selected_region.save(selected_region_path)

            self.root.deiconify()
            self.process_and_display(selected_region_path, index, label)

        canvas.bind("<ButtonPress-1>", on_mouse_down)
        canvas.bind("<B1-Motion>", on_mouse_drag)
        canvas.bind("<ButtonRelease-1>", on_mouse_up)

        selector.mainloop()

    def process_and_display(self, image_path, index, label):
        try:
            original_image, processed_image, mean_processed = self.image_processor.process_image_dbscan(image_path,
                                                                                                        f"processed_image_{index + 1}.png")
            self.display_images(original_image, processed_image, mean_processed, index)

            self.mean_values[label] = mean_processed
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def display_images(self, original_image, processed_image, mean_processed, index):
        if original_image is not None and processed_image is not None:
            original_image_pil = Image.fromarray(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
            original_image_tk = ImageTk.PhotoImage(original_image_pil)
            self.left_frames[index].configure(image=original_image_tk)
            self.left_frames[index].image = original_image_tk
            processed_image_pil = Image.fromarray(processed_image)
            processed_image_tk = ImageTk.PhotoImage(processed_image_pil)

            self.right_frames[index].configure(image=processed_image_tk)
            self.right_frames[index].image = processed_image_tk
            self.mean_labels[index].configure(text=f"Feature grey: {round(mean_processed, 2)}")

    def calculate_bap(self):
        if 'BaP' in self.mean_values:
            mean_value = self.mean_values['BaP']
            result = np.exp(self.model_builder.model_bap.predict([[mean_value]]))
            self.result['BaP'] = result
            print(self.result)

    def calculate_dbp(self):
        if 'DBP' in self.mean_values:
            mean_value = self.mean_values['DBP']
            result = np.exp(self.model_builder.model_dbp.predict([[mean_value]]))
            self.result['DBP'] = result
            print(self.result)

    def calculate_bap_all(self):
        try:
            Bap_T = self.bap_t_var.get()
            input_data = np.array([[Bap_T]])
            bap_light = self.model_builder.model_bap_all.predict(input_data)[0]
            if 'BaP' in self.result and self.result['BaP']:
                bap_img = self.result['BaP'][0]

                weight_light_mse = 1 / self.model_builder.mse_light_bap
                weight_img_mse = 1 / self.model_builder.mse_img_bap

                weight_light_r2 = self.model_builder.r2_light_bap
                weight_img_r2 = self.model_builder.r2_img_bap

                total_weight_mse = weight_light_mse + weight_img_mse
                weight_light_mse_norm = weight_light_mse / total_weight_mse
                weight_img_mse_norm = weight_img_mse / total_weight_mse

                total_weight_r2 = weight_light_r2 + weight_img_r2
                weight_light_r2_norm = weight_light_r2 / total_weight_r2
                weight_img_r2_norm = weight_img_r2 / total_weight_r2

                final_weight_light = (weight_light_mse_norm + weight_light_r2_norm) / 2
                final_weight_img = (weight_img_mse_norm + weight_img_r2_norm) / 2

                weighted_sum_final = (final_weight_light * bap_light) + (final_weight_img * bap_img)
                bap_forecast_final = weighted_sum_final / (final_weight_light + final_weight_img)
                bap_forecast = bap_forecast_final
            else:
                print("no self.result['BaP']")
                bap_forecast = bap_light

            label_str = 'SAFE'
            label_color = 'green'
            if bap_forecast > 10:
                label_str = 'WARNING！'
                label_color = 'red'

            self.bap_forecast_result_label.config(text=f"The calculated BaP result is {bap_forecast:.4f} (ng/mL)")
            self.bap_warning_label.config(text=label_str, fg=label_color)

            self.forecast['BaP'] = bap_forecast
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred while calculating BaP forecast: {e}")

    def calculate_dbp_all(self):
        try:
            DBP_T = self.dbp_t_var.get()
            input_data = np.array([[DBP_T]])
            dbp_light = self.model_builder.model_dbp_all.predict(input_data)[0]
            if 'DBP' in self.result and self.result['DBP']:
                dbp_img = self.result['DBP'][0]

                weight_light_mse = 1 / self.model_builder.mse_light_dbp
                weight_img_mse = 1 / self.model_builder.mse_img_dbp

                weight_light_r2 = self.model_builder.r2_light_dbp
                weight_img_r2 = self.model_builder.r2_img_dbp

                total_weight_mse = weight_light_mse + weight_img_mse
                weight_light_mse_norm = weight_light_mse / total_weight_mse
                weight_img_mse_norm = weight_img_mse / total_weight_mse

                total_weight_r2 = weight_light_r2 + weight_img_r2
                weight_light_r2_norm = weight_light_r2 / total_weight_r2
                weight_img_r2_norm = weight_img_r2 / total_weight_r2

                final_weight_light = (weight_light_mse_norm + weight_light_r2_norm) / 2
                final_weight_img = (weight_img_mse_norm + weight_img_r2_norm) / 2

                weighted_sum_final = (final_weight_light * dbp_light) + (final_weight_img * dbp_img)
                dbp_forecast_final = weighted_sum_final / (final_weight_light + final_weight_img)
                dbp_forecast = dbp_forecast_final
            else:
                print("no self.result['DBP']")
                dbp_forecast = dbp_light

            label_str = 'SAFE'
            label_color = 'green'
            if dbp_forecast > 300:
                label_str = 'WARNING！'
                label_color = 'red'

            self.dbp_forecast_result_label.config(text=f"The calculated DBP result is {dbp_forecast:.4f} (ng/mL)")
            self.dbp_warning_label.config(text=label_str, fg=label_color)

            self.forecast['DBP'] = dbp_forecast
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred while calculating DBP forecast: {e}")

    def reset(self):
        self.mean_values.clear()
        self.result.clear()

        for frame in self.left_frames:
            frame.configure(image=None)
            frame.image = None
        for frame in self.right_frames:
            frame.configure(image=None)
            frame.image = None

        for label in self.mean_labels:
            label.configure(text="Feature grey: ")

        self.bap_forecast_result_label.config(text="Please calculate Bap.")
        self.bap_warning_label.config(text="")
        self.dbp_forecast_result_label.config(text="Please calculate DBP.")
        self.dbp_warning_label.config(text="")


if __name__ == "__main__":
    root = tk.Tk()
    app = ImageProcessorApp(root)
    root.mainloop()