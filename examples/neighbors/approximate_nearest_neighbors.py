import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def find_season_with_highest_sales(file_path):
    # Load the dataset
    data = pd.read_excel(file_path)

    # Encode categorical columns
    data_encoded = pd.get_dummies(data, columns=['Giới Tính', 'Danh Mục', 'Màu Sắc', 'Mùa', 'Tình Trạng Đăng Ký',
                                                  'Phương Thức Thanh Toán', 'Loại Giao Hàng', 'Áp Dụng Giảm Giá',
                                                  'Sử Dụng Mã Khuyến Mãi', 'Phương Thức Thanh Toán Ưu Tiên',
                                                  'Tần Suất Mua Sắm'], drop_first=True)

    # Separate features and target
    X = data_encoded.drop(columns=['Mùa_Xuân', 'Mùa_Hạ', 'Mùa_Thu', 'Mùa_Đông'])
    y = data['Mùa']

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a Random Forest Classifier
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Độ chính xác của mô hình: {accuracy * 100:.2f}%")

    # Predict the season with the highest sales
    season_sales = data['Mùa'].value_counts()
    top_season = season_sales.idxmax()
    top_sales = season_sales.max()

    return top_season, top_sales

if __name__ == "__main__":
    # Prompt user to input the file path
    file_path = input("Nhập đường dẫn tới file Excel: ")

    try:
        # Find the season with the highest sales
        season, sales = find_season_with_highest_sales(file_path)

        # Output the result
        print(f"Mùa có lượng mua lớn nhất là: {season} với {sales} giao dịch.")
    except Exception as e:
        print(f"Đã xảy ra lỗi: {e}")
