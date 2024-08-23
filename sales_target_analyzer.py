import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt  # Import this to fix the NameError

class LogisticRegressionApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Sales Target Analyser")
        self.geometry("600x700")

        # UI Elements

        # Input for sales target
        self.target_frame = ttk.Frame(self)
        self.target_frame.pack(pady=10)
        self.target_label = ttk.Label(self.target_frame, text="Enter Sales Target:")
        self.target_label.pack(side=tk.LEFT)
        self.target_entry = ttk.Entry(self.target_frame)
        self.target_entry.pack(side=tk.LEFT)

        self.train_button = ttk.Button(self, text="Train Model", command=self.train_model)
        self.train_button.pack(pady=10)

        self.results_button = ttk.Button(self, text="Show Results", command=self.show_results)
        self.results_button.pack(pady=10)

        self.plot_button = ttk.Button(self, text="Plot Decision Boundary", command=self.plot_decision_boundary)
        self.plot_button.pack(pady=10)

        self.performance_button = ttk.Button(self, text="View Performance Categories", command=self.show_performance_categories)
        self.performance_button.pack(pady=10)

        self.results_text = tk.Text(self, height=10, width=70)
        self.results_text.pack(pady=10)

        # Entry fields for user input
        self.entries = {}
        for feature in ['Ave Stock Qty', 'Ave Price']:
            frame = ttk.Frame(self)
            frame.pack(pady=5)
            label = ttk.Label(frame, text=f"Enter {feature}:")
            label.pack(side=tk.LEFT)
            entry = ttk.Entry(frame)
            entry.pack(side=tk.LEFT)
            self.entries[feature] = entry

        self.predict_button = ttk.Button(self, text="Predict", command=self.predict_custom_input)
        self.predict_button.pack(pady=10)

        self.prediction_text = tk.Text(self, height=2, width=70)
        self.prediction_text.pack(pady=10)

        # Variables for storing model and data
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.sales_target = None

        # Load the dataset
        self.load_dataset()

    def load_dataset(self):
        # Load your dataset from an Excel file
        df = pd.read_excel('C:/Users/Frank/Desktop/sample_dataset.xlsx')

        # Specify the feature columns and the target column
        feature_columns = ['Ave Stock Qty', 'Ave Price']
        self.sales_column = 'Sales'

        self.X = df[feature_columns]
        self.df = df

    def train_model(self):
        # Get the sales target from the input field
        try:
            self.sales_target = float(self.target_entry.get())
        except ValueError:
            messagebox.showerror("Input Error", "Please enter a valid number for the sales target.")
            return

        if self.sales_target is None:
            messagebox.showerror("Input Error", "Please enter a sales target.")
            return

        # Create the binary target column based on the sales target
        self.y = (self.df[self.sales_column] > self.sales_target).astype(int)

        # Split the data into training and testing sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.3, random_state=42)

        # Train the logistic regression model
        self.model = LogisticRegression()
        self.model.fit(self.X_train, self.y_train)

        messagebox.showinfo("Training Complete", "Model has been trained successfully!")

    def show_results(self):
        if self.model is None:
            messagebox.showerror("Error", "Model is not trained yet!")
            return

        # Make predictions and evaluate the model
        y_pred = self.model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        report = classification_report(self.y_test, y_pred)
        conf_matrix = confusion_matrix(self.y_test, y_pred)

        # Determine performance feedback based on accuracy
        if accuracy == 1.0:
            performance = "Excellent! The model is perfectly predicting the outcomes."
        elif accuracy >= 0.75:
            performance = "Good performance. The model is making mostly correct predictions."
        elif accuracy >= 0.50:
            performance = "Fair performance. The model is making some correct predictions, but there is room for improvement."
        elif accuracy >= 0.33:
            performance = "Poor performance. The model is struggling to make accurate predictions."
        else:
            performance = "Very poor performance. The model is not performing well at all."

        # Display the results and performance feedback in the Text widget
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, f"Accuracy: {accuracy:.2f}\n\n")
        self.results_text.insert(tk.END, "Classification Report:\n")
        self.results_text.insert(tk.END, f"{report}\n")
        self.results_text.insert(tk.END, "Confusion Matrix:\n")
        self.results_text.insert(tk.END, f"{conf_matrix}\n")
        self.results_text.insert(tk.END, f"\nModel Performance: {performance}")

    def plot_decision_boundary(self):
        if self.model is None:
            messagebox.showerror("Error", "Model is not trained yet!")
            return

        # Use the two features for visualization purposes
        X_train_2d = self.X_train
        model_2d = LogisticRegression()
        model_2d.fit(X_train_2d, self.y_train)

        x_min, x_max = X_train_2d.iloc[:, 0].min() - 1, X_train_2d.iloc[:, 0].max() + 1
        y_min, y_max = X_train_2d.iloc[:, 1].min() - 1, X_train_2d.iloc[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                             np.arange(y_min, y_max, 0.1))
        Z = model_2d.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        plt.contourf(xx, yy, Z, alpha=0.3)
        plt.scatter(X_train_2d.iloc[:, 0], X_train_2d.iloc[:, 1], c=self.y_train, edgecolors='k', marker='o')
        plt.xlabel('Ave Stock Qty')
        plt.ylabel('Ave Price')
        plt.title("Decision Boundary")
        plt.show()

    def predict_custom_input(self):
        if self.model is None:
            messagebox.showerror("Error", "Model is not trained yet!")
            return

        input_data = []
        for feature in ['Ave Stock Qty', 'Ave Price']:
            user_input = self.entries[feature].get()
            if not user_input.strip():  # Check if the field is empty
                messagebox.showerror("Input Error", f"Please enter a valid number for {feature}.")
                return
            try:
                value = float(user_input)
                input_data.append(value)
            except ValueError:
                messagebox.showerror("Input Error", f"Please enter a valid number for {feature}.")
                return

        # Convert input data to DataFrame with the correct feature names
        input_df = pd.DataFrame([input_data], columns=['Ave Stock Qty', 'Ave Price'])

        # Make prediction
        prediction = self.model.predict(input_df)[0]

        # Define what each class means
        class_meanings = {
            0: f"Sales below or equal to {self.sales_target}",
            1: f"Sales exceed {self.sales_target}"
        }

        # Get the meaning of the predicted class
        result = class_meanings.get(prediction, f"Class {prediction}: Unknown")

        # Display the prediction result
        self.prediction_text.delete(1.0, tk.END)
        self.prediction_text.insert(tk.END, f"Predicted Result: {result}")

    def show_performance_categories(self):
        # Create a new window for performance categories
        performance_window = tk.Toplevel(self)
        performance_window.title("Performance Categories")
        performance_window.geometry("400x250")

        # Add text describing the performance categories
        text = (
            "Performance Categories:\n\n"
            "Excellent: When accuracy is 1.0 (100%), indicating perfect predictions.\n"
            "Good: When accuracy is between 0.75 and 0.99, showing that the model is performing well.\n"
            "Fair: When accuracy is between 0.50 and 0.74, indicating that the model is performing decently but could be improved.\n"
            "Poor: When accuracy is between 0.33 and 0.49, suggesting the model is not performing well.\n"
            "Very Poor: When accuracy is below 0.33, meaning the model is barely making correct predictions and needs significant improvement."
        )

        label = tk.Label(performance_window, text=text, justify=tk.LEFT)
        label.pack(pady=20, padx=20)

# Running the application
if __name__ == "__main__":
    app = LogisticRegressionApp()
    app.mainloop()
