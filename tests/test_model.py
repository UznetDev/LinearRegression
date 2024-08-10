from linear_regression import LinearRegression

# Ma'lumotlar to'plami
# X - mustaqil o'zgaruvchilar (inputlar)
X = [[1, 2], [2, 3], [3, 4], [4, 5]]

# y - bog'liq o'zgaruvchi (target yoki natija)
y = [30, 50, 70, 90]

# LinearRegression modelini yaratyapmiz
# step - o'rganish sur'ati, n_iters - gradient pasayishining iteratsiya soni
model = LinearRegression(step=0.01, n_iters=1000)

# Modelni o'qitamiz
# fit funksiyasi X va y ma'lumotlarini ishlatib, slope va intercept qiymatlarini hisoblaydi
model.fit(X, y)

# Yangi ma'lumotlar uchun oldindan bashorat qilish (predict funksiyasi yordamida)
y_pred = model.predict(X)

# Natijalarni chop etamiz
# Bu yerda modelning slope (qiyalik) va intercept (kesishuv nuqtasi) qiymatlarini ko'rishimiz mumkin
print("Slope:", model._slope_)           # Slope - qiyalik (β1, β2, ... βn)
print("Intercept:", model._intercept_)   # Intercept - kesishuv nuqtasi (β0)

# MSE - O'rtacha kvadrat xatolik (Mean Squared Error) qiymatini hisoblaymiz
print("MSE:", model.MSE(y, y_pred))

# Predict - Oldindan bashorat qilingan qiymatlarni ko'rsatamiz
print("Predict:", y_pred)
