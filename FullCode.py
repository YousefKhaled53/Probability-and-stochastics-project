import sys
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QLabel, QLineEdit, QPushButton, QFormLayout, \
    QWidget, QFileDialog, QInputDialog, QMessageBox
import matplotlib
from scipy.io import loadmat
from scipy.integrate import simps
from mpl_toolkits.mplot3d import Axes3D

matplotlib.use('Qt5Agg')


def calculate_mean(sample_space):
    if len(sample_space) == 0:
        raise ValueError("Empty sample space")

    return sum(sample_space) / len(sample_space)

def calculate_variance(sample_space):
    if len(sample_space) == 0:
        raise ValueError("Empty sample space")

    mean_val = calculate_mean(sample_space)
    squared_diff = [(x - mean_val) ** 2 for x in sample_space]

    return sum(squared_diff) / len(sample_space)

def calculate_third_moment(sample_space):
    if len(sample_space) == 0:
        raise ValueError("Empty sample space")

    mean_val = calculate_mean(sample_space)
    var_val = calculate_variance(sample_space)
    n = len(sample_space)

    # Calculate the numerator and denominator
    skewness = np.sum(((sample_space - mean_val)/(np.sqrt(var_val))) ** 3) / n
    return skewness

def calc_mgf_t0(sample_space):
    t=0
    mgf_values = sum(np.exp(t * xi) for xi in sample_space) / len(sample_space)
    return mgf_values

def calc_mgf_prime_t0(sample_space):
    t=0
    mgf_prime_values = sum(xi * np.exp(t * xi) for xi in sample_space) / len(sample_space)
    return mgf_prime_values

def calc_mgf_prime2_t0(sample_space):
    t = 0
    mgf_prime2_values = sum(xi ** 2 * np.exp(t * xi) for xi in sample_space) / len(sample_space)
    return mgf_prime2_values


class StatisticsTool(QMainWindow):
    def __init__(self):
        super().__init__()
        self.sample_space = []  # Initialize as an empty list
        self.time_input = []  # Add this line
        self.amplitude_input = []  # Add this line
        self.time_inputs = []
        self.amplitude_inputs = []
        self.num_sample_functions_input = None
        self.num_sample_functions = None
        self.initUI()

        # Additional attributes to store widgets and layouts
        self.central_widget = None
        self.layout = None
        self.sample_space_input = None
        self.mean_label = None
        self.variance_label = None
        self.third_moment_label = None
        self.mgf0_label = None
        self.mgf_prime0_label = None
        self.mgf_prime20_label = None
        self.calc_button = None
        self.upload_button = None


    def initUI(self):
        # Central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        self.layout = QVBoxLayout(central_widget)

        # Add buttons for choosing random variable or random process
        self.random_variable_button = QPushButton('Random Variable', self)
        self.random_variable_button.clicked.connect(self.init_random_variable)

        self.random_process_button = QPushButton('Random Process', self)
        self.random_process_button.clicked.connect(self.init_random_process)

        # Add buttons to layout
        self.layout.addWidget(self.random_variable_button)
        self.layout.addWidget(self.random_process_button)

        # Set font and style
        font = self.font()
        font.setPointSize(12)
        self.setFont(font)

        # Apply stylesheets
        self.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #3498db, stop:1 #2980b9);
                color: #ecf0f1;
                border: 1px solid #2980b9;
                padding: 10px 20px;
                border-radius: 5px;
                qproperty-shadow: 0px 2px 5px rgba(0, 0, 0, 0.2);
            }

            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #2980b9, stop:1 #3498db);
                border: 1px solid #2980b9;
                qproperty-shadow: 0px 5px 10px rgba(0, 0, 0, 0.3);
            }
        """)

        # Set window title and show UI
        self.setWindowTitle('Probability project fall 2023')
        self.setGeometry(300, 300, 500, 200)
        self.show()

    def init_random_variable(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # Input field for the sample space
        form_layout = QFormLayout()
        self.sample_space_input = QLineEdit(self)
        form_layout.addRow('Sample Space (comma-separated):', self.sample_space_input)

        # Labels for displaying results
        self.mean_label = QLabel('Mean: N/A', self)
        self.variance_label = QLabel('Variance: N/A', self)
        self.third_moment_label = QLabel('Third Moment: N/A', self)
        # Labels for displaying additional results
        self.mgf0_label = QLabel('MGF at t=0: N/A', self)
        self.mgf_prime0_label = QLabel("MGF' at t=0: N/A", self)
        self.mgf_prime20_label = QLabel("MGF'' at t=0: N/A", self)

        # Add widgets to layout
        layout.addLayout(form_layout)
        layout.addWidget(self.mean_label)
        layout.addWidget(self.variance_label)
        layout.addWidget(self.third_moment_label)
        # Add labels to layout
        layout.addWidget(self.mgf0_label)
        layout.addWidget(self.mgf_prime0_label)
        layout.addWidget(self.mgf_prime20_label)

        # Button to calculate statistics
        self.calc_button = QPushButton('Calculate', self)
        self.calc_button.clicked.connect(self.calculate_statistics)
        layout.addWidget(self.calc_button)

        # Add file upload option
        self.upload_button = QPushButton('Upload File', self)
        self.upload_button.clicked.connect(self.upload_file)
        layout.addWidget(self.upload_button)

        # Set font and style
        font = self.font()
        font.setPointSize(12)
        self.setFont(font)

        # Apply stylesheets
        self.setStyleSheet("""
                    QLabel {
                        font-size: 20px;
                        color: #333;
                    }
                    QPushButton {
                        background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #3498db, stop:1 #2980b9);
                        color: #ecf0f1;
                        border: 1px solid #2980b9;
                        padding: 10px 20px;
                        border-radius: 5px;
                        qproperty-shadow: 0px 2px 5px rgba(0, 0, 0, 0.2);
                    }

                    QPushButton:hover {
                        background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #2980b9, stop:1 #3498db);
                        border: 1px solid #2980b9;
                        qproperty-shadow: 0px 5px 10px rgba(0, 0, 0, 0.3);
                    }
                    QLineEdit {
                        font-size: 20px;
                        padding: 5px;
                        border: 1px solid #ccc;
                        border-radius: 5px;
                    }
                """)

        # Set window title and show UI
        self.setWindowTitle('Probability project fall 2023')
        self.setGeometry(300, 300, 500, 400)
        self.show()

    def init_random_process(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        self.layout = QVBoxLayout(central_widget)

        # Input field for the number of sample functions
        form_layout = QFormLayout()
        self.num_sample_functions_input = QLineEdit(self)
        form_layout.addRow('Number of Sample Functions youu are going to enter:', self.num_sample_functions_input)

        # Set font for the input field
        font = self.font()
        font.setPointSize(12)
        self.num_sample_functions_input.setFont(font)

        # Add form layout to the main layout
        self.layout.addLayout(form_layout)

        # Add button to create input fields for sample functions
        self.create_input_fields_button = QPushButton('Create Input Fields', self)
        self.create_input_fields_button.clicked.connect(self.create_input_fields)
        self.layout.addWidget(self.create_input_fields_button)

        # Add file upload option
        self.upload_button = QPushButton('Upload MATLAB File', self)
        self.upload_button.clicked.connect(self.upload_matlab_file)
        self.layout.addWidget(self.upload_button)

        # Set font for the buttons
        self.random_variable_button.setFont(font)
        self.random_process_button.setFont(font)
        self.create_input_fields_button.setFont(font)
        self.upload_button.setFont(font)

        # Set window title and show UI
        self.setWindowTitle('Probability project fall 2023 - Random Process')
        self.setGeometry(300, 300, 600, 600)
        self.show()

    def upload_matlab_file(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly

        file_path, _ = QFileDialog.getOpenFileName(self, "Open MATLAB File", "", "MATLAB Files (*.mat);;All Files (*)",
                                                   options=options)

        if file_path:
            try:
                time_variable_name, ok = QInputDialog.getText(self, 'Time Variable Name',
                                                              'Enter the name of the time variable:')
                if not ok:
                    return

                amplitude_variable_name, ok = QInputDialog.getText(self, 'Amplitude Variable Name',
                                                                   'Enter the name of the amplitude variable:')
                if not ok:
                    return

                data = loadmat(file_path)
                if time_variable_name in data and amplitude_variable_name in data:
                    self.time_inputs = np.array(data[time_variable_name][0])
                    self.amplitude_inputs = np.array(data[amplitude_variable_name])

                    # Set the number of sample functions input
                    self.num_sample_functions_input.setText(str(self.amplitude_inputs.shape[0]))

                    # Create input fields with pre-filled data
                    self.create_input_fields()

                    QMessageBox.information(self, 'File Loaded', 'MATLAB file loaded successfully.')
                else:
                    QMessageBox.warning(self, 'Error', 'One or both variables not found in the MATLAB file.')
            except Exception as e:
                QMessageBox.warning(self, 'Error', f'Error during file loading: {e}')

    def create_input_fields(self):
        try:
            self.num_sample_functions = int(self.num_sample_functions_input.text())
            if self.num_sample_functions <= 0:
                QMessageBox.warning(self, 'Error',
                                    'Please enter a valid number greater than zero for the number of sample functions.')
                return
            # Clear existing layout
            self.clear_layout(self.layout)

            self.time_inputs_list = []
            self.amplitude_inputs_list = []
            form_layout = QFormLayout()

            # Add input field for the time vector
            time_input = QLineEdit(self)
            time_label = QLabel(f'Time Vector for Function (comma-separated):', self)
            if(len(self.time_inputs) != 0):
                time_input.setText(', '.join(map(str, self.time_inputs)))
            self.time_inputs_list.append(time_input)
            self.plot_no = QLineEdit(self)
            form_layout.addRow('M of Sample Functions you want to plot:', self.plot_no)
            # Add input field for the user to enter the index of the amplitude function for time mean calculation
            self.amplitude_index_input = QLineEdit(self)
            form_layout.addRow('Amplitude Function Index for Time Mean and time acf:', self.amplitude_index_input)
            form_layout.addRow(time_label, time_input)


            # Add label to display the result

            for i in range(1, self.num_sample_functions + 1):
                # Use lists instead of individual variables
                amplitude_input = QLineEdit(self)
                amplitude_label = QLabel(f'Amplitude Vector for Function {i} (comma-separated):', self)
                form_layout.addRow(amplitude_label, amplitude_input)
                if (len(self.amplitude_inputs) != 0):
                    amplitude_input.setText(', '.join(map(str, self.amplitude_inputs[i - 1])))
                self.layout.addLayout(form_layout)

                # Append inputs to the lists
                self.amplitude_inputs_list.append(amplitude_input)

            # Add button to perform calculations
            # Add label for displaying the result
            self.time_mean_result_label = QLabel('Time Mean Result: N/A', self)
            self.time_acf_reslut_label = QLabel('Time ACF Result : N/A' , self)
            self.Total_average_label = QLabel('Total Power Average : N/A', self)

            self.layout.addWidget(self.time_mean_result_label)
            self.layout.addWidget(self.time_acf_reslut_label)
            self.layout.addWidget(self.Total_average_label)


            self.calculate_button = QPushButton('Calculate', self)
            self.calculate_button.clicked.connect(self.calculate_random_process)
            self.layout.addWidget(self.calculate_button)

        except ValueError:
            QMessageBox.warning(self, 'Error', 'Please enter a valid number for the number of sample functions.')

        except ValueError:
            QMessageBox.warning(self, 'Error', 'Please enter a valid number for the number of sample functions.')

    def calculate_random_process(self):
        try:
            # Check if the number of sample functions to plot is greater than the total number of sample functions
            num_functions_to_plot = int(self.plot_no.text())
            if num_functions_to_plot > self.num_sample_functions:
                QMessageBox.warning(self, 'Error',
                                    'Number of sample functions to plot cannot exceed the total number of sample functions.')
                return
            # Clear existing layout
            time_arrays = []
            amplitude_arrays = []
            time_inputs_str = self.time_inputs_list[0].text().strip()
            self.time_inputs = np.array(list(map(float, time_inputs_str.split(','))))
            time_arrays.append(self.time_inputs)
            # Use a flag to check for errors
            error_occurred = False

            for i in range(self.num_sample_functions):
                amplitude_inputs_str = self.amplitude_inputs_list[i].text().strip()
                self.amplitude_inputs = np.array(list(map(float, amplitude_inputs_str.split(','))))
                amplitude_arrays.append(self.amplitude_inputs)

                # Check if the number of amplitude values is equal to the number of time values
                if len(self.amplitude_inputs) != len(self.time_inputs):
                    QMessageBox.warning(
                        self,
                        'Error',
                        f'Number of amplitude values for Sample Function {i + 1} does not match the number of time values.'
                    )
                    error_occurred = True
                    break  # Exit the loop if there's a mismatch

            #  =================== ENSEMBLE MEAN  ===================
            ensemble_mean = []
            for i in range(0, len(self.time_inputs)):
                mean = 0
                for j in range(self.num_sample_functions):
                    mean += amplitude_arrays[j][i]
                fin_value = mean / self.num_sample_functions
                ensemble_mean.append(fin_value)
            #--------------------------------------------------------

            # =================== TOTAL AVERAGE POWER  ===================
            # Iterate through each sample function
            total_average_power = 0.0
            num_elements = 0
            for sample_function in amplitude_arrays:
                # Iterate through each amplitude value in the sample function
                for amplitude_value in sample_function:
                    # Accumulate the squared magnitude
                    total_average_power += amplitude_value ** 2
                    num_elements += 1

            # Check for division by zero
            if num_elements > 0:
                total_average_power /= num_elements
            else:
                print("Warning: Empty sample space, total average power is undefined.")

            self.Total_average_label.setText(f'Total Average Power Result: {total_average_power}')
            #--------------------------------------------------------


            # Get user input for the index of the amplitude function for time mean calculation
            plot_no_index = int(self.plot_no.text())

            # =================== TIME MEAN  ===================

            # Calculate the time mean using the logic from the random process mean calculation
            time_mean_result = self.calculate_time_mean(amplitude_arrays)

            # Display the result on the page
            self.time_mean_result_label.setText(f'Time Mean Result: {time_mean_result}')
            #--------------------------------------------------------
            # =================== TIME ACF  ===================
            time_acf_result = self.calculate_time_acf(amplitude_arrays)

            # Display the result on the page
            self.time_acf_reslut_label.setText(f'Time ACF Result: {time_acf_result}')

            #--------------------------------------------------------

            # =================== Plotting  ===================
            if not error_occurred:
                fig, axs = plt.subplots(plot_no_index + 1, 1, figsize=(6, 2 * (plot_no_index + 1)))

                # Plot sample functions
                for i in range(plot_no_index):
                    ax = axs[i]  # Use a separate axis for each sample function
                    ax.set_title(f'Sample Function {i + 1}')
                    ax.plot(time_arrays[0], amplitude_arrays[i])
                    ax.set_xlabel('Time')
                    ax.set_ylabel('Amplitude')

                # Plot ensemble mean
                ax = axs[plot_no_index]  # Use the last row for ensemble mean
                ax.set_title('Ensemble Mean of the Process')
                ax.plot(self.time_inputs, ensemble_mean)
                ax.set_xlabel('Time')
                ax.set_ylabel('Amplitude')

                # Remove empty subplots
                for i in range(plot_no_index + 1, axs.shape[0]):
                    axs[i].axis('off')  # Turn off the axis for the empty subplots

                # Plot autocorrelation function
                autocorrelation_function = self.calculate_autocorrelation(amplitude_arrays)

                i, j = np.meshgrid(self.time_inputs, self.time_inputs)

                # Assuming self.calculate_autocorrelation returns the autocorrelation values
                autocorrelation_function = self.calculate_autocorrelation(amplitude_arrays)
                fig = plt.figure(figsize=(10, 8))
                ax = fig.add_subplot(111, projection='3d')
                surface = ax.plot_surface(i, j, autocorrelation_function, cmap='viridis', rstride=1, cstride=1,
                                          linewidth=0, antialiased=False)
                ax.set_title('Autocorrelation Function of the Process')
                ax.set_xlabel('(i)')
                ax.set_ylabel('(j)')
                ax.set_zlabel('Autocorrelation')
                # Add colorbar
                cbar = fig.colorbar(surface, ax=ax, shrink=0.5, aspect=10)
                cbar.set_label('Autocorrelation Value')
                self.calculate_psd(autocorrelation_function)
        except ValueError:
            QMessageBox.warning(self, 'Error', 'Please enter a valid number for the number of sample functions to plot.')
        except Exception as e:
            print(f"Error during random process calculation: {e}")

    def calculate_psd(self , acf):
        N = acf.shape[0]

        # Sampling frequency (assuming unit sampling for simplicity)
        fs = 1

        # Calculate the frequencies corresponding to the Fourier transform
        frequencies = np.fft.fftshift(np.fft.fftfreq(N, 1 / fs))

        # Calculate the power spectral density using the Fourier transform of the ACF
        psd = np.abs(np.fft.fftshift(np.fft.fft(acf)))

        # Plot the PSD
        plt.figure()
        plt.plot(frequencies, psd)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power/Frequency')
        plt.title('Power Spectral Density')
        plt.grid(True)
        plt.show()


    def calculate_time_acf(self, amplitude_arrays):
        try:
            amplitude_index = int(self.amplitude_index_input.text()) - 1  # Subtract 1 to convert to zero-based index

            if amplitude_index < 0 or amplitude_index >= len(amplitude_arrays):
                QMessageBox.warning(self, 'Error', 'Invalid amplitude index.')
                return

            # Get the amplitude array for the selected index
            amplitude_array = amplitude_arrays[amplitude_index]
            M = len(amplitude_arrays[0])  # Number of time values
            time_acf_result = 0

            for i in range(M):
                for j in range(i,M):
                    S = amplitude_array[i] * amplitude_array[j]
                    time_acf_result = time_acf_result + S
            time_acf_result = time_acf_result / self.time_inputs[-1]
        except ValueError:
            QMessageBox.warning(self, 'Error', 'Please enter a valid amplitude index.')
        return time_acf_result

    def calculate_time_mean(self, amplitude_arrays):
        try:
            amplitude_index = int(self.amplitude_index_input.text()) - 1  # Subtract 1 to convert to zero-based index

            if amplitude_index < 0 or amplitude_index >= len(amplitude_arrays):
                QMessageBox.warning(self, 'Error', 'Invalid amplitude index.')
                return

            # Get the amplitude array for the selected index
            amplitude_array = amplitude_arrays[amplitude_index]
            time_interval = self.time_inputs[-1] - self.time_inputs[0]

            # Shift the time values to be centered around zero
            shifted_time_inputs = self.time_inputs - time_interval / 2

            # Perform integration using the trapezoidal rule
            #time_mean_result = simps(amplitude_array, shifted_time_inputs)
            time_mean_result = sum(amplitude_array) / time_interval
            # Display the result on the page

        except ValueError:
            QMessageBox.warning(self, 'Error', 'Please enter a valid amplitude index.')
        return time_mean_result

    def calculate_autocorrelation(self, amplitude_arrays):
        M = len(amplitude_arrays[0])  # Number of time values
        N = len(self.time_inputs)
        Transposed_amp = np.transpose(amplitude_arrays)
        ensemble_autocorrelation = np.zeros((M, M))

        for i in range(M):
            for j in range(M):
                S = np.dot(Transposed_amp[i] , Transposed_amp[j])
                acf = np.sum(S)
                ensemble_autocorrelation[i][j] = acf / len(Transposed_amp[0])
        return ensemble_autocorrelation

    def clear_layout(self, layout):
        if layout is not None:
            while layout.count():
                item = layout.takeAt(0)
                if item.widget():
                    item.widget().deleteLater()
                elif item.layout():
                    self.clear_layout(item.layout())

    def calculate_statistics(self):
        try:
            if len(self.sample_space) == 0:
                sample_space_str = self.sample_space_input.text().strip()
                if not sample_space_str:
                    raise ValueError("No valid numeric values in the sample space.")
                try:
                    sample_space = np.array(list(map(float, sample_space_str.split(','))))
                except ValueError:
                    raise ValueError("Invalid input. Please enter only numeric values.")
            else:
                sample_space = self.sample_space

            print(len(sample_space))
            print(f"Data type of loaded array: {sample_space.dtype}")

            if len(sample_space) == 0:
                print("Error: Empty sample space.")
                return

            mean = calculate_mean(sample_space)
            variance = calculate_variance(sample_space)
            third_moment = calculate_third_moment(sample_space)
            mgf0 = calc_mgf_t0(sample_space)
            mgf_prime = calc_mgf_prime_t0(sample_space)
            mgf_prime2 = calc_mgf_prime2_t0(sample_space)

            self.mean_label.setText(f'Mean: {mean}')
            self.variance_label.setText(f'Variance: {variance}')
            self.third_moment_label.setText(f'Third Moment: {third_moment}')
            self.mgf0_label.setText(f'MGF at t=0: {mgf0}')
            self.mgf_prime0_label.setText(f'MGF\' at t=0: {mgf_prime}')
            self.mgf_prime20_label.setText(f'MGF\'\' at t=0: {mgf_prime2}')

            self.plot_mgf_and_derivatives(sample_space)
        except ValueError as ve:
            QMessageBox.warning(self, 'Error', str(ve))
        except Exception as e:
            print(f"Error during calculation: {e}")

    def upload_file(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly

        file_path, _ = QFileDialog.getOpenFileName(self, "Open MATLAB File", "", "MATLAB Files (*.mat);;All Files (*)",options=options)

        if file_path:
            try:
                variable_name, ok = QInputDialog.getText(self, 'Variable Name', 'Enter variable name:')
                if ok:
                    data = loadmat(file_path)
                    print(len(data))
                    if variable_name in data:
                        self.sample_space = np.array(data[variable_name][0])
                        print(f"Loaded array shape: {self.sample_space.shape}")
                        print(f"Loaded array size: {self.sample_space.size}")
                        sample_space = self.sample_space.flatten()
                        print(f"Flattened array shape: {self.sample_space.shape}")
                        print(f"Flattened array size: {self.sample_space.size}")
                        self.sample_space_input.setText(', '.join(map(str, sample_space)))
                        print(len(self.sample_space))
                        QMessageBox.information(self, 'File Loaded', 'File loaded successfully.')
                    else:
                        QMessageBox.warning(self, 'Error', 'Variable not found in the MATLAB file.')
            except Exception as e:
                QMessageBox.warning(self, 'Error', f'Error during file loading: {e}')
        print(len(self.sample_space))

    def plot_mgf_and_derivatives(self, sample_space):
        t_values = np.linspace(0, 2, 100)
        mgf_values = [sum(np.exp(t_val * xi) for xi in sample_space) / len(sample_space) for t_val in t_values]
        mgf_prime_values = [sum(xi * np.exp(t_val * xi) for xi in sample_space) / len(sample_space) for t_val in t_values]
        mgf_prime2_values = [sum(xi ** 2 * np.exp(t_val * xi) for xi in sample_space) / len(sample_space) for t_val in t_values]

        plt.figure(figsize=(12, 6))

        plt.subplot(1, 3, 1)
        plt.plot(t_values, mgf_values, label='MGF M(t)')
        plt.title('MGF M(t)')
        plt.xlabel('t')
        plt.ylabel('MGF M(t)')
        plt.legend()

        plt.subplot(1, 3, 2)
        plt.plot(t_values, mgf_prime_values, label="M'(t)")
        plt.title("First Derivative of M(t)")
        plt.xlabel('t')
        plt.ylabel('(M\'(t)')
        plt.legend()

        plt.subplot(1, 3, 3)
        plt.plot(t_values, mgf_prime2_values, label="M''(t)")
        plt.title("Second Derivative of M(t)")
        plt.xlabel('t')
        plt.ylabel('M\'\'(t)')
        plt.legend()

        plt.tight_layout()
        plt.show()

def main():
    app = QApplication(sys.argv)
    ex = StatisticsTool()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()