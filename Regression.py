import pandas as pd
import matplotlib.pyplot as plt
import re
from sklearn.model_selection import train_test_split
from unidecode import unidecode
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
from lazypredict.Supervised import LazyRegressor
from lightgbm import LGBMRegressor
from sklearn.model_selection import GridSearchCV

data = pd.read_csv("VN_housing_dataset.csv")

# xóa cột và hàng dư thừa
data = data.drop("Unnamed: 0", axis=1)
data = data.drop([data.index[82496]])

# kiểm tra từng Feature
'''
 0   Ngày             Thiếu 0
 1   Địa chỉ          Thiếu 47
 2   Quận             thiếu 1
 3   Huyện            Thiếu 47
 4   Loại hình nhà ở  Thiếu 31
 5   Giấy tờ pháp lý  Thiếu 28886
 6   Số tầng          Thiếu 46097
 7   Số phòng ngủ     Thiếu 38
 8   Diện tích        Thiếu 1
 9   Dài vs Rộng      Thiếu 46929            
 11  Giá/m2           Thiếu 12

'''
# tìm xem có bao nhiêu hàng thiếu cả dài lẫn rộng
filtered_Dai_Rong = data.loc[data['Dài'].isna() & data['Rộng'].isna() & data['Diện tích'].isna()]
two_nan = filtered_Dai_Rong.index.tolist()
print(len(two_nan))

# xử lý Huyện
# Huyện xẽ được thay bằng Quận trong hàng tương ứng
data['Huyện'].fillna(data['Quận'], inplace=True)

# xử lý Quận
# xóa hàng không có thông tin quận và huyện
data.dropna(subset=['Quận'], inplace=True)

# xử lý cột địa chỉ
data["Tỉnh/Thành phố"] = "Hà Nội"
def fill_missing_address(row):
    huyen = row['Huyện']
    quan = row['Quận']
    thanh_pho = row['Tỉnh/Thành phố']

    if pd.notnull(huyen) and pd.notnull(quan):
        return huyen + ', ' + quan + ', ' + thanh_pho
    elif pd.notnull(huyen):
        return huyen + ', ' + thanh_pho
    elif pd.notnull(quan):
        return quan + ', ' + thanh_pho
    else:
        return thanh_pho


data['Địa chỉ'] = data.apply(fill_missing_address, axis=1)

data['Quận_tableau'] = data['Quận'].str.replace('Thị xã', '').str.replace('Huyện', '').str.replace('Quận', '')

data['Quận_tableau'] = data['Quận_tableau'].apply(lambda x: unidecode(x).strip())
data['Quận_tableau'] = data['Quận_tableau'].replace(['Nam Tu Liem', 'Bac Tu Liem'], 'Tu Liem')

# xử lý giá
# xóa những hàng không có giá
data.dropna(subset=['Giá/m2'], inplace=True)

# xử lý loại hình nhà ở
data["Loại hình nhà ở"].fillna("Không rõ", inplace=True)

# xử lý giấy tờ pháp lý
data['Giấy tờ pháp lý'].fillna("Không rõ", inplace=True)

# xử lý số tầng
data['Số tầng'].fillna(1, inplace=True)

# xử lý số phòng ngủ
data['Số phòng ngủ'].fillna('1', inplace=True)

# bỏ hai cột dài và rộng
data = data.drop(columns=["Dài", "Rộng"])
# chuyển kiểu dữ liệu của từng cột
data['Ngày'] = pd.to_datetime(data['Ngày'])

# xem các giá trị có thể trong cột số tầng
# cập nhật giá trị của cột số tầng chuyển số tầng nhiều hơn 10 thành 11
# đổi kiểu dữ liệu của cột này luôn
data.loc[data['Số tầng'] == 'Nhiều hơn 10', 'Số tầng'] = 11
data['Số tầng'] = data['Số tầng'].astype(int)

# xem các giá trị có thể trong cột số phòng ngủ
# cập nhật giá trị của cột số phòng ngủ chuyển số phòng ngủ nhiều hơn 10 thành 11
# đổi kiểu dữ liệu của cột này luôn
data.loc[data["Số phòng ngủ"] == 'nhiều hơn 10 phòng', 'Số phòng ngủ'] = '11'
data["Số phòng ngủ"] = data["Số phòng ngủ"].str.replace('phòng', '')
data["Số phòng ngủ"] = data["Số phòng ngủ"].astype(int)

# xem các giá trị có thể trong cột diện tích
# cập nhật giá trị của cột diện tích
# đổi kiểu dữ liệu của cột này luôn
data["Diện tích"] = data["Diện tích"].str.replace("m²", '')
data["Diện tích"] = data["Diện tích"].astype(float)

# thêm cột đơn vị cho giá/m2
# xem các giá trị có thể trong cột giá
# cập nhật giá trị của cột giá
# đổi kiểu dữ liệu của cột này luôn
data['Đơn vị'] = data['Giá/m2'].apply(lambda x: re.findall(r'[^\d.,]+', x)[0])

data['Giá/m2'] = data['Giá/m2'].str.replace('.', '')
data['Giá/m2'] = data['Giá/m2'].str.replace(',', '.')
data["Giá/m2"] = data["Giá/m2"].str.replace("triệu/m²", '').str.replace("đ/m²", '').str.replace("tỷ/m²", '')

data["Giá/m2"] = data["Giá/m2"].astype(float)

def convert_price(row):
    if row['Đơn vị'] == ' tỷ/m²':
        return row['Giá/m2'] * 1000
    else:
        return row["Giá/m2"]

data["Giá/m2"] = data.apply(convert_price, axis=1)

data = data.drop(columns="Đơn vị")
data = data.rename(columns={'Giá/m2': 'Giá (triệu đồng/m2)'})

# tạo table mới
selected_columns = ["Số tầng", "Số phòng ngủ", "Diện tích", "Giá (triệu đồng/m2)"]
new_data = data[selected_columns].copy()

# loại bỏ các outlier
# cột diện tích
min_ = data['Diện tích'].quantile(0.001)
max_ = data['Diện tích'].quantile(0.999)

data = data[data['Diện tích'] >= min_]
data = data[data['Diện tích'] <= max_]

# plt.boxplot(data["Diện tích"])
# plt.show()
# cột giá
data['Lower Bound'] = data.groupby(['Quận'])['Giá (triệu đồng/m2)'].transform(lambda x: x.quantile(0.25) - 1.5 * (x.quantile(0.75) - x.quantile(0.25)))
data['Upper Bound'] = data.groupby(['Quận'])['Giá (triệu đồng/m2)'].transform(lambda x: x.quantile(0.75) + 1.5 * (x.quantile(0.75) - x.quantile(0.25)))

data = data[(data['Giá (triệu đồng/m2)'] >= data['Lower Bound']) & (data['Giá (triệu đồng/m2)'] <= data['Upper Bound'])]
data = data.drop(['Lower Bound', 'Upper Bound'], axis=1)

# cột số tầng
data = data.drop(data[data['Số tầng'] > 10].index)

# tạo data_new
columns_to_drop = ["Ngày", "Quận_tableau", "Địa chỉ", "Tỉnh/Thành phố", "Huyện"]
data_new = data.drop(columns=columns_to_drop)

# phân chia bộ dữ liệu
target = "Giá (triệu đồng/m2)"
x = data_new.drop(target, axis=1)
y = data[target]
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.2, random_state=42)

# tạo khung biến đổi dữ liệu về cùng range
# sử lý dữ liệu dạng Numerical
num_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="mean")),
    ("encoder", StandardScaler())
])

# sử lý dữ liệu dạng Categorical
# nominal
# ordinal
kieunha_value = ['Không rõ', 'Nhà ngõ, hẻm', 'Nhà phố liền kề', 'Nhà mặt phố, mặt tiền', 'Nhà biệt thự']
kieugiayto_value = ['Đã có sổ', 'Không rõ', 'Đang chờ sổ', 'Giấy tờ khác']
kieuquan_value = ["Huyện Ba Vì", 'Huyện Thạch Thất', 'Thị xã Sơn Tây', 'Huyện Mê Linh', 'Huyện Thường Tín', 'Huyện Sóc Sơn', 'Huyện Chương Mỹ', 'Huyện Quốc Oai',
                   'Huyện Đông Anh', 'Huyện Mỹ Đức', 'Huyện Phúc Thọ', 'Huyện Đan Phượng', 'Huyện Thanh Oai', 'Huyện Hoài Đức', 'Huyện Gia Lâm' , 'Huyện Thanh Trì',
                  'Quận Long Biên', 'Quận Hà Đông', 'Huyện Phú Xuyên', 'Quận Hoàng Mai', 'Quận Nam Từ Liêm', 'Quận Bắc Từ Liêm', 'Quận Hai Bà Trưng',
                  'Quận Thanh Xuân', 'Quận Tây Hồ', 'Quận Đống Đa', 'Quận Cầu Giấy', 'Quận Ba Đình', 'Quận Hoàn Kiếm']

ord_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OrdinalEncoder(categories=[kieugiayto_value, kieunha_value, kieuquan_value]))
])

preprocessor = ColumnTransformer(transformers=[
    ("num_feature", num_transformer, ["Số tầng", "Số phòng ngủ", "Diện tích"]),
    ("ord_feature", ord_transformer, ["Giấy tờ pháp lý", "Loại hình nhà ở", "Quận"])
])

# tạo model
reg = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", LGBMRegressor())
])

param_grid = {
    "regressor__boosting_type": ["gbdt", "dart", "rf"],
    "regressor__learning_rate": [0.1, 0.2, 0.3],
    "regressor__n_estimators": [100, 200, 300]
}
reg_cv = GridSearchCV(reg, param_grid=param_grid, scoring="r2", verbose=2, cv=5, n_jobs=8)
# train model
reg_cv.fit(x_train, y_train)
y_pridict = reg_cv.predict(x_test)

# sử dụng metrics để đánh giá
print(r2_score(y_test, y_pridict))
print(mean_absolute_error(y_test, y_pridict))
for i, j in zip(y_test, y_pridict):
    print("Thực tế: {}, Dự đoán: {}".format(i, j))

# test = pd.DataFrame([["Quận Hai Bà Trưng", "Nhà ngõ, hẻm", "Đã có sổ", 4, 3, 52]], columns=["Quận", "Loại hình nhà ở", "Giấy tờ pháp lý", "Số tầng", "Số phòng ngủ", "Diện tích"])
# # Create the pandas DataFrame
# y_thu = reg.predict(test)
# print(y_thu)
# biendoi = Pipeline(steps=[
#     ("preprocessor", preprocessor)
# ])

# x_train = biendoi.fit_transform(x_train)
# x_test = biendoi.transform(x_test)
#
# reg_cv = LazyRegressor()
#
# # fitting data in LazyClassifier
# models, predictions = reg_cv.fit(x_train, x_test, y_train, y_test)
#
# print(predictions)
