    - Window the data into sequences.
    - Normalize features.
    """
    df = pd.read_csv("C:\Users\Pranjal\Downloads\ezyZip.zip")
    df = df.sort_values(by=['employee_id', 'timestamp'])  # Ensure chronological order
    
    # Extract features (acc_x, acc_y, acc_z) and labels (employee_id)
    X = df[['acc_x', 'acc_y', 'acc_z']].values
    y = df['employee_id'].values  # 1 to 30
    
    # Window the data: Split into overlapping windows of size 'window_size'
    windows = []
    labels = []
    for emp_id in np.unique(y):
        emp_data = X[y == emp_id]
        for i in range(0, len(emp_data) - window_size, window_size // 2):  # Overlap for better sequences
            window = emp_data[i:i + window_size]
            if len(window) == window_size:  # Ensure full window
                windows.append(window)
                labels.append(emp_id)
    
    X_windows = np.array(windows)
    y_labels = np.array(labels)
    
    # Normalize the data
    scaler = StandardScaler()
    X_windows = scaler.fit_transform(X_windows.reshape(-1, 3)).reshape(-1, window_size, 3)
    
    return X_windows, y_labels, scaler
def add_noise(data, noise_factor=0.01):
    """Data augmentation: Add Gaussian noise for robustness."""
    noise = np.random.randn(*data.shape) * noise_factor
    return data + noise






Models
