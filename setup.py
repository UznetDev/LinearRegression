from setuptools import setup, find_packages

setup(
    name='linear_regression',
    version='1.0.0',
    description='A simple implementation of linear regression using gradient descent in Python',
    author='UZNet Dev',
    author_email='uznetdev@example.com',
    url='https://github.com/uznetdev/LinearRegression',
    packages=find_packages(),  # Loyihadagi barcha paketlarni avtomatik ravishda topadi
    install_requires=[],  # Agar kerak bo'lsa, bu yerga boshqa kutubxonalarni qo'shing, masalan ['numpy', 'scipy']
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)

