from setuptools import setup

setup(name='datamunger',
    version='0.3.3',
    description = 'Set of tools for Nan and Outlier Imputation',
    url = 'https://github.com/bacross/datamunger',
    author = 'bacross',
    author_email = 'pydatamunger@gmail.com',
    license='MIT',
    packages=['datamunger'],
	install_requires=['numpy','pandas','sklearn','joblib'],
    zip_safe=False)